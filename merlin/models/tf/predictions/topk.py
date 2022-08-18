from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

import merlin.io
from merlin.core.dispatch import DataFrameType
from merlin.models.tf.core.base import Block
from merlin.models.tf.predictions.base import MetricsFn, Prediction, PredictionBlock
from merlin.models.tf.predictions.classification import default_categorical_prediction_metrics
from merlin.models.tf.utils import tf_utils
from merlin.schema import Tags


class TopKPrediction(PredictionBlock):
    """Prediction block for top-k evaluation

    Parameters
    ----------
    item_dataset:  merlin.io.Dataset,
        Dataset of the pretrained item embeddings to use for the top-k index.
    prediction: TopKLayer,
        The layer for indexing the pre-trained candidates and retrieving top-k candidates.
        By default None
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for binary-classification
        by 'binary_crossentropy'
    default_metrics_fn: Callable
        A function returning the list of default metrics
        to use for binary-classification
    """

    def __init__(
        self,
        item_dataset: merlin.io.Dataset,
        prediction: "TopKLayer" = None,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        k: int = 10,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics_fn: MetricsFn = default_categorical_prediction_metrics,
        **kwargs,
    ):
        if prediction is None:
            prediction = BruteForce(k=k)

        prediction = prediction.index_from_dataset(item_dataset)
        super().__init__(
            prediction=prediction,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def compile(self, k=None):
        self.prediction._k = k


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TopKLayer(Layer):
    def __init__(self, k: int, **kwargs) -> None:
        """Initializes the base class."""
        super().__init__(**kwargs)
        self._k = k

    def index(self, candidates: tf.Tensor, identifiers: Optional[tf.Tensor] = None) -> "TopKLayer":
        """Builds the retrieval index.
        When called multiple times the existing index will be dropped and a new one
        created.

        Parameters:
        -----------
            candidates: tensor of candidate embeddings.
            identifiers: Optional tensor of candidate identifiers. If
                given, these will be used as identifiers of top candidates returned
                when performing searches. If not given, indices into the candidates
                tensor will be returned instead.
        Returns:
        Self
        """
        raise NotImplementedError()

    def index_from_dataset(
        self, data: merlin.io.Dataset, check_unique_ids: bool = True, **kwargs
    ) -> "TopKLayer":
        """Builds the retrieval index from a merlin dataset.

        Parameters
        ----------
        data : merlin.io.Dataset
            The dataset with the pre-trained item embeddings
        check_unique_ids : bool, optional
            Whether to check if `data` has unique indices, by default True

        Returns
        -------
        TopKLayer
            return the class with retrieval index
        """
        ids, values = self.extract_ids_embeddings(data, check_unique_ids)
        return self.index(candidates=values, identifiers=ids, **kwargs)

    @staticmethod
    def _check_unique_ids(data: DataFrameType):
        if data.index.to_series().nunique() != data.shape[0]:
            raise ValueError("Please make sure that `data` contains unique indices")

    def extract_ids_embeddings(self, data: merlin.io.Dataset, check_unique_ids: bool = True):
        """Extract tensors of candidates and indices from the merlin dataset

        Parameters
        ----------
        data : merlin.io.Dataset
            The dataset with the pre-trained item embeddings
        check_unique_ids : bool, optional
            Whether to check if `data` has unique indices, by default True
        """
        if hasattr(data, "to_ddf"):
            data = data.to_ddf()
        if check_unique_ids:
            self._check_unique_ids(data=data)
        values = tf_utils.df_to_tensor(data)
        ids = tf_utils.df_to_tensor(data.index)

        if len(ids.shape) == 2:
            ids = tf.squeeze(ids)
        return ids, values

    def get_candidates_dataset(
        self, block: Block, data: merlin.io.Dataset, id_column: Optional[str] = None
    ):
        from merlin.models.tf.utils.batch_utils import TFModelEncode

        if not id_column and getattr(block, "schema", None):
            tagged = block.schema.select_by_tag(Tags.ITEM_ID)
            if tagged.column_schemas:
                id_column = tagged.first.name

        model_encode = TFModelEncode(model=block, output_concat_func=np.concatenate)

        data = data.to_ddf()
        embedding_ddf = data.map_partitions(model_encode, filter_input_columns=[id_column])
        embedding_df = embedding_ddf.compute(scheduler="synchronous")

        embedding_df.set_index(id_column, inplace=True)
        return embedding_df

    def from_block(
        self, block: Block, data: merlin.io.Dataset, id_column: Optional[str] = None, **kwargs
    ):
        """Build candidates embeddings from applying `block` to a dataset of features `data`.

        Parameters:
        -----------
        block: Block
            The Block that returns embeddings from raw item features.
        data: merlin.io.Dataset
            Dataset containing raw item features.
        id_column: Optional[str]
            The candidates ids column name.
            Note, this will be inferred automatically if the block contains
            a schema with an item-id Tag.
        """
        candidates_dataset = self.get_candidates_dataset(block, data, id_column)
        return self.index_from_dataset(candidates_dataset, **kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    def _score(self, queries: tf.Tensor, candidates: tf.Tensor) -> tf.Tensor:
        """Computes the standard dot product score from queries and candidates."""
        return tf.matmul(queries, candidates, transpose_b=True)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class BruteForce(TopKLayer):
    """Brute force retrieval top-k layer."""

    def __init__(self, k: int = 10, name: Optional[str] = None):
        """Initializes the layer.
        Args:
        query_model: Optional Keras model for representing queries. If provided,
            will be used to transform raw features into query embeddings when
            querying the layer. If not provided, the layer will expect to be given
            query embeddings as inputs.
        k: Default k.
        name: Name of the layer.
        """

        super().__init__(k=k, name=name)

        self._candidates = None

    def index(self, candidates: tf.Tensor, identifiers: Optional[tf.Tensor]) -> "BruteForce":

        self._ids = self.add_weight(
            name="ids",
            dtype=tf.int32,
            shape=identifiers.shape,
            initializer=tf.keras.initializers.Constant(value=tf.cast(identifiers, tf.int32)),
            trainable=False,
        )

        self._candidates = self.add_weight(
            name="candidates",
            dtype=tf.float32,
            shape=candidates.shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )

        self._ids.assign(tf.cast(identifiers, tf.int32))
        self._candidates.assign(tf.cast(candidates, tf.float32))
        return self

    def call(self, inputs, targets=None, k=None, *args, **kwargs) -> "Prediction":
        if not k:
            k = self._k
        scores = self._score(inputs, self._candidates)
        top_scores, top_ids = tf.math.top_k(scores, k=k)
        if targets is not None:
            targets = tf.cast(tf.squeeze(targets), tf.int32)
            targets = tf.cast(tf.expand_dims(targets, -1) == top_ids, tf.float32)
            targets = tf.reshape(targets, tf.shape(top_scores))
        return Prediction(top_scores, targets, top_ids=top_ids)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return tf.TensorShape((batch_size, self._k)), tf.TensorShape((batch_size, self._k))
