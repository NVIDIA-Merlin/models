import logging
from typing import Dict, Optional, Sequence, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..block.mlp import MLPBlock
from .base import PredictionTask


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]


LOG = logging.getLogger("merlin_models")


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
        summary_type="first",
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            **kwargs,
        )
        self.pre = tf.keras.layers.Dense(1, activation="sigmoid", name=self.child_name("logit"))


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
        summary_type="first",
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            **kwargs,
        )
        self.pre = tf.keras.layers.Dense(1, name=self.child_name("logit"))
