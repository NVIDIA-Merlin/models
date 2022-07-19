from typing import Optional, Sequence, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.metrics.topk import AvgPrecisionAt, MRRAt, NDCGAt, PrecisionAt, RecallAt
from merlin.models.tf.predictions.base import ContrastivePredictionBlock

# Or: RetrievalCategoricalPrediction
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DotProductCategoricalPrediction(ContrastivePredictionBlock):
    DEFAULT_K = 10

    def __init__(
        self,
        negative_sampling="in-batch",
        downscore_false_negatives=False,
        target: Optional[str] = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical-cross-entropy",
        default_metrics: Sequence[tf.keras.metrics.Metric] = (
            RecallAt(DEFAULT_K),
            MRRAt(DEFAULT_K),
            NDCGAt(DEFAULT_K),
            AvgPrecisionAt(DEFAULT_K),
            PrecisionAt(DEFAULT_K),
        ),
        query_name: str = "query",
        item_name: str = "item",
        **kwargs,
    ):
        super().__init__(
            prediction=DotProduct(query_name, item_name),
            prediction_with_negatives=ContrastiveDotProduct(query_name, item_name),
            default_loss=default_loss,
            default_metrics=default_metrics,
            name=name,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            negative_sampling=negative_sampling,
            downscore_false_negatives=downscore_false_negatives,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DotProduct(Layer):
    def __init__(self, query_name: str = "query", item_name: str = "item", **kwargs):
        super().__init__(**kwargs)
        self.query_name = query_name
        self.item_name = item_name

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(
            tf.multiply(inputs[self.query_name], inputs[self.item_name]), keepdims=True, axis=-1
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def get_config(self):
        return {
            **super(DotProduct, self).get_config(),
            "query_name": self.query_name,
            "item_name": self.item_name,
        }


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ContrastiveDotProduct(DotProduct):
    def __init__(
        self,
        negative_sampling="in-batch",
        downscore_false_negatives=False,
        sampling_downscore_false_negatives_value: float = MIN_FLOAT,
        query_name: str = "query",
        item_name: str = "item",
        **kwargs,
    ):
        super().__init__(query_name, item_name, **kwargs)
        self.negative_sampling = negative_sampling
        self.downscore_false_negatives = downscore_false_negatives
        self.sampling_downscore_false_negatives_value = sampling_downscore_false_negatives_value

    def build(self, input_shape):
        super(DotProduct, self).build(input_shape)
        self.item_id_feature_name = self.schema.select_by_tag(Tags.ITEM_ID).first.name

    def call(self, inputs, features, targets, **kwargs):
        outputs = inputs

        positive_scores = super(ContrastiveDotProduct, self).call(outputs, **kwargs)
        positive_item_ids = features[self.item_id_feature_name]

        if isinstance(targets, tf.Tensor) and len(targets.shape) == len(outputs.shape) - 1:
            outputs = tf.squeeze(outputs)

        return Prediction(outputs, targets)
