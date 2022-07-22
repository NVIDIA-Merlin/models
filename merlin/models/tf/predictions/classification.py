from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.metrics.topk import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.models.tf.predictions.base import ContrastivePredictionBlock, PredictionBlock
from merlin.models.tf.predictions.sampling.base import ItemSamplersType
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import ColumnSchema, Schema


class BinaryPrediction(PredictionBlock):
    """Binary-classification prediction block"""

    def __init__(
        self,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss="binary_crossentropy",
        default_metrics=(
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
        ),
        **kwargs,
    ):
        super().__init__(
            prediction=tf.keras.layers.Dense(1, activation="sigmoid"),
            default_loss=default_loss,
            default_metrics=default_metrics,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            name=name,
            **kwargs,
        )


class CategoricalPrediction(ContrastivePredictionBlock):
    """Categorical prediction block"""

    DEFAULT_K = 10

    def __init__(
        self,
        target: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ],
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss="categorical_crossentropy",
        default_metrics: Sequence[tf.keras.metrics.Metric] = (
            RecallAt(DEFAULT_K),
            MRRAt(DEFAULT_K),
            NDCGAt(DEFAULT_K),
            AvgPrecisionAt(DEFAULT_K),
            PrecisionAt(DEFAULT_K),
        ),
        **kwargs,
    ):
        if isinstance(target, (Schema, ColumnSchema)):
            prediction = CategoricalTarget(target)
        elif isinstance(target, EmbeddingTable):
            prediction = EmbeddingTablePrediction(target)
            pass
        else:
            prediction = target

        super().__init__(
            self,
            prediction=prediction,
            prediction_with_negatives=ContrastiveLookUps(prediction),
            default_loss=default_loss,
            default_metrics=default_metrics,
            name=name,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )


class CategoricalTarget(tf.keras.layers.Dense):
    """Prediction of a categorical column."""

    def __init__(
        self,
        feature: Union[Schema, ColumnSchema],
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
        if isinstance(feature, Schema):
            assert len(feature) == 1, "Schema can have max 1 feature"
            col_schema = feature.first
        else:
            col_schema = feature

        self.num_classes = col_schema.int_domain.max + 1
        units = kwargs.pop("units", self.num_classes)

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

    def embedding_lookup(self, inputs, **kwargs):
        return embedding_ops.embedding_lookup(tf.transpose(self.kernel), inputs, **kwargs)


class EmbeddingTablePrediction(Layer):
    """Prediction using weight-sharing with an embedding table"""

    def __init__(self, table: EmbeddingTable, bias_initializer="zeros", **kwargs):
        self.table = table
        self.nun_classes = table.input_dim

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


# TODO
class ContrastiveLookUps(Layer):
    def __init__(
        self,
        prediction: LookUpProtocol,
        negative_samplers: ItemSamplersType = "popularity",
        downscore_false_negatives=True,
        false_negative_score: float = MIN_FLOAT,
    ):
        self.prediction = prediction
