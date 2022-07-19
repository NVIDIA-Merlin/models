from typing import Optional, Sequence, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.predictions.base import ContrastivePredictionBlock


# Or: RetrievalCategoricalPrediction
class DotProductCategoricalPrediction(ContrastivePredictionBlock):
    def __init__(
        self,
        negative_sampling="in-batch",
        downscore_false_negatives=False,
        target=None,
        pre=None,
        post=None,
        logits_temperature=1.0,
        name=None,
        default_loss="categorical-cross-entropy",
        default_metrics=(),
        default_contrastive_metrics=(),
        **kwargs,
    ):
        super().__init__(
            prediction=DotProduct(),
            prediction_with_negatives=ContrastiveDotProduct(),
            default_loss=default_loss,
            default_metrics=default_metrics,
            default_contrastive_metrics=default_contrastive_metrics,
            name=name,
            target=target,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            negative_sampling=negative_sampling,
            downscore_false_negatives=downscore_false_negatives,
            **kwargs,
        )


class DotProduct(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class ContrastiveDotProduct(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
