#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
from typing import List, Optional, Protocol, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.metrics.topk import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.models.tf.predictions.base import ContrastivePredictionBlock, MetricsFn, PredictionBlock
from merlin.models.tf.predictions.sampling.base import (
    Items,
    ItemSamplersType,
    parse_negative_samplers,
)
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    call_layer,
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import ColumnSchema, Schema

LOG = logging.getLogger("merlin_models")


def default_binary_metrics():
    return (
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    )


def default_categorical_prediction_metrics(k=10):
    return (
        RecallAt(k),
        MRRAt(k),
        NDCGAt(k),
        AvgPrecisionAt(k),
        PrecisionAt(k),
    )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BinaryPrediction(PredictionBlock):
    """
    Binary-classification prediction block.

    Parameters
    ----------
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
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy",
        default_metrics_fn: MetricsFn = default_binary_metrics,
        **kwargs,
    ):
        prediction = kwargs.pop("prediction", None)
        super().__init__(
            prediction=prediction or tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalPrediction(ContrastivePredictionBlock):
    """Categorical prediction block

    Parameters
    ----------
    prediction: Union[Schema, ColumnSchema,
                EmbeddingTable, 'CategoricalTarget',
                'EmbeddingTablePrediction']
        The target feature to predict. To perform weight-tying [1] technique, you should provide
        the `EmbeddingTable` or `EmbeddingTablePrediction` related to the
        target feature.
    negative_samplers: ItemSamplersType, optional
        List of samplers for negative sampling,
        by default None
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1
    name: str, optional
        The name of the task, by default None
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for categorical-classification
        by default 'categorical_crossentropy'
    get_default_metrics: Callable, optional
        A function returning the list of default metrics
        to use for categorical-classification

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).

    """

    def __init__(
        self,
        prediction: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ] = None,
        negative_samplers: ItemSamplersType = None,
        target_name: str = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics_fn: MetricsFn = default_categorical_prediction_metrics,
        **kwargs,
    ):
        self.max_num_samples = kwargs.pop("max_num_samples", None)
        _prediction = kwargs.pop("prediction", None)

        if prediction is not None:
            if isinstance(prediction, (Schema, ColumnSchema)):
                _prediction = CategoricalTarget(prediction)
                if isinstance(prediction, Schema):
                    prediction = prediction.first
                target_name = target_name or prediction.name
            elif isinstance(prediction, EmbeddingTable):
                _prediction = EmbeddingTablePrediction(prediction)
                target_name = _prediction.table.col_schema.name
            else:
                _prediction = prediction

        prediction_with_negatives = kwargs.pop(
            "prediction_with_negatives",
            SampledLookUps(
                prediction=_prediction,
                negative_samplers=negative_samplers,
                feature_name=target_name,
                max_num_samples=self.max_num_samples,
            ),
        )
        self.target_name = kwargs.pop("target", target_name)
        super().__init__(
            prediction=_prediction,
            prediction_with_negatives=prediction_with_negatives,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=self.target_name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def compile(self, negative_sampling=None, downscore_false_negatives=False):
        if negative_sampling is not None:
            negative_sampling = parse_negative_samplers(negative_sampling)
        self.prediction_with_negatives.negative_samplers = negative_sampling
        self.prediction_with_negatives.downscore_false_negatives = downscore_false_negatives

    def get_config(self):
        config = super(ContrastivePredictionBlock, self).get_config()
        config["max_num_samples"] = self.max_num_samples
        config["target_name"] = self.target_name
        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalTarget(tf.keras.layers.Dense):
    """Prediction of a categorical feature.

    Parameters
    ----------
    feature : Union[Schema, ColumnSchema]
        The schema description of the categorical feature to predict.
    activation : optional
        Activation function to use, by default None
    use_bias : bool, optional
        Whether the layer uses a bias vector, by default True
    kernel_initializer : str, optional
        Initializer for the kernel weights matrix, by default "glorot_uniform"
    bias_initializer : str, optional
        Initializer for the bias vector., by default "zeros"
    kernel_regularizer : optional
        Regularizer function applied to the kernel weights matrix, by default None
    bias_regularizer : optional
        Regularizer function applied to the bias vector, by default None
    activity_regularizer : optional
        Regularizer function applied to the output of the layer (its "activation"),
        by default None
    kernel_constraint : optional
        Constraint function applied to the kernel weights matrix, by default None
    bias_constraint : optional
        Constraint function applied to the bias vector, by default None
    """

    def __init__(
        self,
        feature: Union[Schema, ColumnSchema] = None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        if feature is not None:
            if isinstance(feature, Schema):
                assert len(feature) == 1, "Schema can have max 1 feature"
                col_schema = feature.first
            else:
                col_schema = feature

            self.num_classes = col_schema.int_domain.max + 1
            units = self.num_classes
        else:
            units = kwargs.pop("units")
            self.num_classes = units

        super().__init__(
            units,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )

    def embedding_lookup(self, inputs: tf.Tensor, **kwargs):
        """Method to retrieve hidden representation vectors from the kernel weight matrix
        based on a given "input" positions

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of indices to retrieve from the weight matrix.

        Returns
        -------
        tf.Tensor
            Tensor of hidden representation vectors.
        """
        return embedding_ops.embedding_lookup(tf.transpose(self.kernel), inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class EmbeddingTablePrediction(Layer):
    """Prediction of a categorical feature using weight-sharing [1] with an embedding table

    Parameters
    ----------
    table : EmbeddingTable
        The embedding table to use as the weight matrix
    bias_initializer : str, optional
        Initializer for the bias vector, by default "zeros"

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).
    """

    def __init__(self, table: EmbeddingTable, bias_initializer="zeros", **kwargs):
        self.table = table
        self.num_classes = table.input_dim
        self.bias_initializer = bias_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )

        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        logits = tf.matmul(inputs, self.table.table.embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

    def embedding_lookup(self, inputs, **kwargs):
        return self.table.table(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def get_config(self):
        config = maybe_serialize_keras_objects(
            self,
            {
                **super().get_config(),
            },
            ["table"],
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["table"])
        return super().from_config(config)


@runtime_checkable
class LookUpProtocol(Protocol):
    def embedding_lookup(self, inputs, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SampledLookUps(Layer):
    """Contrastive layer for sampled logits.

    This layer can be used to compute the output scores of a multi-classification
    task only on a subset of sampled classes.

    For example, we could use this class to define the sampled softmax task [1] where
    negatives are defined using a popularity-based sampler.

    Parameters
    ----------
    prediction : LookUpProtocol
        The prediction layer used for computing the logits scores. It should be an
        instance of `LookUpProtocol`, i.e. it includes the method `embedding_lookup`
        that indexes the output weights.
    negative_samplers : ItemSamplersType
        List of samplers for negative sampling,
    feature_name : str, optional
        The name of the target feature, by default None
    downscore_false_negatives : bool, optional
        Identify false negatives (sampled item ids equal to the positive item and downscore them
        to the `sampling_downscore_false_negatives_value`),
        by default False
    false_negative_score : float, optional
        Value to be used to downscore false negatives when
        `sampling_downscore_false_negatives=True`,
        by default `np.finfo(np.float32).min / 100.0`

    References:
    -----------
    [1] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312
    """

    def __init__(
        self,
        prediction: LookUpProtocol,
        negative_samplers: ItemSamplersType,
        feature_name: str = None,
        downscore_false_negatives=True,
        false_negative_score: float = MIN_FLOAT,
        **kwargs,
    ):
        self.prediction = prediction
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.feature_name = feature_name

        if negative_samplers is not None:
            self.negative_samplers = parse_negative_samplers(negative_samplers)
        else:
            self.negative_samplers = negative_samplers

        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, features, targets, training=False, testing=False):
        if isinstance(targets, dict):
            if self.feature_name is None:
                raise ValueError(
                    "When training with multi-task, you should specify the "
                    "`target_name` for the sampled softmax task"
                )
            targets = targets[self.feature_name]
        # Get positive weights
        pos_item_id = tf.squeeze(targets)
        positive_weights = self.prediction.embedding_lookup(pos_item_id)

        # Sample negative items
        neg_items = self.sample_negatives(
            Items(pos_item_id, {}), features, training=training, testing=testing
        )
        negative_weights = self.prediction.embedding_lookup(neg_items.id)

        # Apply dot-product
        negative_scores = tf.linalg.matmul(inputs, negative_weights, transpose_b=True)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs, positive_weights), keepdims=True, axis=-1
        )

        if self.downscore_false_negatives:
            negative_scores, _ = rescore_false_negatives(
                pos_item_id, neg_items.id, negative_scores, self.false_negative_score
            )

        outputs = tf.concat([positive_scores, negative_scores], axis=-1)

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        outputs = tf.cast(outputs, tf.float32)
        outputs = tf.squeeze(outputs)

        targets = tf.concat(
            [
                tf.ones([tf.shape(outputs)[0], 1], dtype=outputs.dtype),
                tf.zeros(
                    [tf.shape(outputs)[0], tf.shape(outputs)[1] - 1],
                    dtype=outputs.dtype,
                ),
            ],
            axis=1,
        )

        return Prediction(outputs, targets)

    def sample_negatives(
        self,
        positive_items: Items,
        features: TabularData,
        training=False,
        testing=False,
    ) -> Items:
        """Method to sample negatives from `self.negative_samplers`

        Parameters
        ----------
        positive_items : Items
            Positive items (ids and metadata)
        features : TabularData
            Dictionary of input raw tensors
        training : bool, optional
            Flag for train mode, by default False
        testing : bool, optional
            Flag for test mode, by default False

        Returns
        -------
        Items
            Class containing the sampled negative ids
        """
        negative_items: List[Items] = []
        sampling_kwargs = {"training": training, "testing": testing, "features": features}

        # sample a number of negative ids from self.negative_samplers
        for sampler in self.negative_samplers:
            sampler_items: Items = call_layer(sampler, positive_items, **sampling_kwargs)

            if tf.shape(sampler_items.id)[0] > 0:
                negative_items.append(sampler_items)
            else:
                LOG.warn(
                    f"The sampler {type(sampler).__name__} returned no samples for this batch."
                )

        if len(negative_items) == 0:
            raise Exception(f"No negative items where sampled from samplers {self.samplers}")

        negatives = negative_items[0]
        if len(negative_items) > 1:
            for neg in negative_items[1:]:
                negatives += neg

        return negatives

    @property
    def has_negative_samplers(self) -> bool:
        return self.negative_samplers is not None and len(self.negative_samplers) > 0

    def get_config(self):
        config = maybe_serialize_keras_objects(
            self,
            {
                **super().get_config(),
                "downscore_false_negatives": self.downscore_false_negatives,
                "false_negative_score": self.false_negative_score,
                "num_classes": self.num_classes,
            },
            ["negative_samplers", "prediction"],
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["negative_samplers", "prediction"])
        return super().from_config(config)
