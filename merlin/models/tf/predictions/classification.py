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
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.metrics.topk import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.models.tf.predictions.base import ContrastivePredictionBlock, PredictionBlock
from merlin.models.tf.predictions.sampling.base import Items, ItemSamplersType, ItemSamplerV2
from merlin.models.tf.predictions.sampling.popularity import PopularityBasedSampler
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
    default_metrics: Sequence[tf.keras.metrics.Metric], optional
        Default metrics to use for binary-classification
    """

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy",
        default_metrics=(
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ),
        **kwargs,
    ):
        prediction = kwargs.pop("prediction", None)
        super().__init__(
            prediction=prediction or tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics=default_metrics,
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
    target : Union[Schema, ColumnSchema,
                EmbeddingTable, 'CategoricalTarget&quot',
                'EmbeddingTablePrediction']
        The target feature to predict. To perform weight-tying [1] technique, you should provide
        the `EmbeddingTable` or `EmbeddingTablePrediction` related to the
        taregt feature.
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
        The name of the task., by default None
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for categorical-classification
        by default 'categorical_crossentropy'
    default_metrics: Sequence[tf.keras.metrics.Metric], optional
        Default metrics to use categorical-classification

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).

    """

    DEFAULT_K = 10

    def __init__(
        self,
        target: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ] = None,
        target_name: str = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics: Sequence[tf.keras.metrics.Metric] = (
            RecallAt(DEFAULT_K),
            MRRAt(DEFAULT_K),
            NDCGAt(DEFAULT_K),
            AvgPrecisionAt(DEFAULT_K),
            PrecisionAt(DEFAULT_K),
        ),
        **kwargs,
    ):
        self.target_name = target_name
        self.max_num_samples = kwargs.pop("max_num_samples", None)
        prediction = kwargs.pop("prediction", None)
        if target is not None:
            if isinstance(target, (Schema, ColumnSchema)):
                prediction = CategoricalTarget(target)
            elif isinstance(target, EmbeddingTable):
                prediction = EmbeddingTablePrediction(target)
            else:
                prediction = target

        prediction_with_negatives = kwargs.pop(
            "prediction_with_negatives",
            ContrastiveLookUps(
                prediction, feature_name=target_name, max_num_samples=self.max_num_samples
            ),
        )
        super().__init__(
            prediction=prediction,
            prediction_with_negatives=prediction_with_negatives,
            default_loss=default_loss,
            default_metrics=default_metrics,
            name=name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

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
        self.nun_classes = table.input_dim
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )

        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        logits = tf.matmul(inputs, self.table.embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

    def embedding_lookup(self, inputs, **kwargs):
        return self.table(inputs, **kwargs)


@runtime_checkable
class LookUpProtocol(Protocol):
    def embedding_lookup(self, inputs, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContrastiveLookUps(Layer):
    def __init__(
        self,
        prediction: LookUpProtocol,
        feature_name: str = None,
        negative_samplers: ItemSamplersType = None,
        downscore_false_negatives=True,
        false_negative_score: float = MIN_FLOAT,
        **kwargs,
    ):
        self.prediction = prediction
        self.num_classes = prediction.num_classes
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.feature_name = feature_name

        if negative_samplers is None:
            negative_samplers = PopularityBasedSampler(self.num_classes, **kwargs)
        if not isinstance(negative_samplers, (list, tuple)):
            negative_samplers = [negative_samplers]
        self.negative_samplers = [ItemSamplerV2.parse(s) for s in list(negative_samplers)]
        assert (
            len(self.negative_samplers) > 0
        ), "At least one sampler is required by ContrastiveLookUps for negative sampling"
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, features, targets, training=False, testing=False):
        if isinstance(targets, dict):
            assert self.feature_name is not None, "When training multi-task, you should specify the"
            f" the target name `{self.feature_name}` of for the sampled softmax task"
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
        positive_scores = tf.linalg.matmul(inputs, positive_weights, transpose_b=True)

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
            Class containing sampled negative ids
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

        negatives = sum(negative_items) if len(negative_items) > 1 else negative_items[0]

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
