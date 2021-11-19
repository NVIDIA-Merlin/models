from typing import List, Optional, Sequence, Text, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..core import Block, LossBlock, PredictionTask, MetricOrMetricClass


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy()
    DEFAULT_METRICS = (
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.AUC,
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS,
        **kwargs,
    ):
        super().__init__(
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )
        self.logit = tf.keras.layers.Dense(1, activation="sigmoid", name=self.child_name("logit"))
        self.loss = loss

    def _compute_loss(self, predictions, targets, sample_weight=None, training: bool = False,
                      **kwargs) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return self.logit(inputs)

