from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..core import PredictionTask


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RegressionTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.MeanSquaredError()
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        **kwargs,
    ):
        super().__init__(
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )
        self.logit = tf.keras.layers.Dense(1, name=self.child_name("logit"))
        self.loss = loss

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return self.logit(inputs)
