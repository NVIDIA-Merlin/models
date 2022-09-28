from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

import merlin.io
from merlin.core.dispatch import DataFrameType
from merlin.models.tf.blocks.core import Block, block_registry
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.outputs.base import MetricsFn, ModelOutput
from merlin.models.tf.outputs.classification import default_categorical_prediction_metrics
from merlin.models.tf.utils import tf_utils
from merlin.schema import Schema


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TopKLayer(Layer):
    """Top-K index to retrieve top-k scores and indices from an item block.

    Parameters:
    -----------
    k: int
        The cut-off for top-k retrieval
    """

    def __init__(self, k: int, **kwargs) -> None:
        """Initializes the base class."""
        super().__init__(**kwargs)
        self._k = k

    def index(self, candidates: tf.Tensor, identifiers: Optional[tf.Tensor] = None) -> "TopKLayer":
        """Set the index used for retrieving the top-k candidates.
        When called multiple times the existing index will be dropped and a new one
        created.
        Parameters:
        -----------
            candidates: tensor of candidate embeddings.
            identifiers: Optional tensor of candidate identifiers. If
                given, these will be used as identifiers of top candidates returned
                when performing searches. If not given, indices into the candidates
                tensor will be returned instead.

        """
        raise NotImplementedError()

    def index_from_dataset(
        self, data: merlin.io.Dataset, check_unique_ids: bool = True
    ) -> "TopKLayer":
        """Builds the top-k retrieval index from a merlin dataset.
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
        return self.index(candidates=values, identifiers=ids)

    @staticmethod
    def _check_unique_ids(data: DataFrameType):
        if data.index.to_series().nunique() != data.shape[0]:
            raise ValueError("Please make sure that `data` contains unique indices")

    def extract_ids_embeddings(self, data: merlin.io.Dataset, check_unique_ids: bool = True):
        """Extract tensors of candidates and indices from the merlin dataset
        Parameters
        ----------
        data : merlin.io.Dataset
            The dataset with the pre-trained candidates embeddings,
            indexed by the candidate's id column
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

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Method to return the tuple of top-k (ids, scores)"""
        raise NotImplementedError()

    def _score(self, queries: tf.Tensor, candidates: tf.Tensor) -> tf.Tensor:
        """Computes the standard dot product score from queries and candidates."""
        return tf.matmul(queries, candidates, transpose_b=True)


@Block.registry.register("brute-force-topk")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BruteForce(TopKLayer):
    """Top-k layer that performs Brute force search over the candidates index."""

    def __init__(self, k: int = 10, name: Optional[str] = None):
        """Initializes the layer.

        Parameters:
        -----------
        k: int
            Default threshold for top-k retrieval. By default 10
        name: Optional[str]
            Name of the layer. By default None
        """

        super().__init__(k=k, name=name)

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

    def call(
        self, inputs, targets=None, k=None, testing=False
    ) -> Union[Prediction, Tuple[tf.Tensor, tf.Tensor]]:
        """Compute the scores between the query inputs and all indexed candidates,
        then retrieve the top-k candidates with the highest scores.

        Parameters
        ----------
        inputs : tf.Tensor
            The query embeddings representation
        targets: tf.Tensor
            The tensor of positive candidates
        k : int, optional
            The threshold for top-k retrieval, by default None
        testing: bool
        """
        if not k:
            k = self._k
        scores = self._score(inputs, self._candidates)
        top_scores, top_ids = tf.math.top_k(scores, k=k)
        if testing:
            assert targets is not None, ValueError(
                "Targets should be provided during evaluation mode"
            )
            targets = tf.cast(tf.squeeze(targets), tf.int32)
            targets = tf.cast(tf.expand_dims(targets, -1) == top_ids, tf.float32)
            targets = tf.reshape(targets, tf.shape(top_scores))
            return Prediction(top_scores, targets)
        return top_scores, top_ids


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TopKOutput(ModelOutput):
    """Prediction block for top-k evaluation
    Parameters
    ----------
    to_call:  Union[str, tf.keras.layers.Layer, TopKLayer]
       The top-k layer to use for retrieving top-k candidates ids and scores
    item_dataset:  merlin.io.Dataset,
        Dataset of the pretrained candidates embeddings to use for the top-k index.
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
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for binary-classification
        by 'categorical_crossentropy'
    default_metrics_fn: Callable
        A function returning the list of default metrics
        to use for ranking evaluation
    """

    def __init__(
        self,
        to_call: Union[str, TopKLayer],
        candidate_dataset: merlin.io.Dataset = None,
        k: int = 10,
        target: Optional[str, Schema] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics_fn: MetricsFn = default_categorical_prediction_metrics,
        **kwargs,
    ):

        if isinstance(to_call, str):
            if candidate_dataset is None:
                raise ValueError(
                    "You should provide the dataset of pre-trained embeddings"
                    " when `to_call` is the string name of the top-k strategy "
                )
                to_call = block_registry.parse(to_call).index_from_dataset(candidate_dataset)

        super().__init__(
            to_call=to_call,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )
