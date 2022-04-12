from typing import Dict, List, Union

import tensorflow as tf
from tensorflow.python.keras.metrics import Metric

from merlin.models.tf.blocks.core.base import PredictionOutput
from merlin.models.tf.utils.mixins import MetricsMixin


class RetrievalMetrics(MetricsMixin):
    def __init__(self, metrics: List[Metric]):
        super().__init__()
        self._metrics = metrics

    def calculate_metrics(
        self,
        outputs: PredictionOutput,
        mode: str = "val",
        forward: bool = True,
        training: bool = False,
        **kwargs
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        pass

    def metric_results(self, mode: str = "val") -> Dict[str, Union[float, tf.Tensor]]:
        pass

    def reset_metrics(self):
        pass
