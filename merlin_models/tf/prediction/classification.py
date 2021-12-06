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

from typing import Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from merlin_standard_lib import Schema, Tag

from ..core import MetricOrMetricClass, PredictionBlock, PredictionTask
from .ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt


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

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return self.logit(inputs)


class Softmax(PredictionBlock):
    def __init__(
        self,
        schema: Schema,
        feature_name: str = Tag.ITEM_ID,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        **kwargs,
    ):
        super(Softmax, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.num_classes = schema.categorical_cardinalities()[feature_name]
        self.feature_name = feature_name

    def build(self, input_shape):
        self.output_layer = Dense(
            units=self.num_classes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.feature_name}-softmax",
            activation="softmax",
        )
        return super().build(input_shape)

    def predict(self, inputs, targets=None, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = self.output_layer(inputs)
        predictions = tf.nn.log_softmax(logits, axis=-1)
        return predictions, targets

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_classes,)


class MultiClassClassificationTask(PredictionTask):
    DEFAULT_LOSS = SparseCategoricalCrossentropy(from_logits=True)
    DEFAULT_METRICS = {
        "ranking": (NDCGAt([10, 20]), RecallAt([10, 20]), AvgPrecisionAt([10, 20])),
        "multi-class": (),
    }

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS["multi-class"],
        pre_call: Optional[PredictionBlock] = None,
        pre_loss: Optional[PredictionBlock] = None,
        **kwargs,
    ):

        super().__init__(
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre_call=pre_call,
            pre_loss=pre_loss,
            **kwargs,
        )
        self.loss = loss

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        feature_name: str = Tag.ITEM_ID,
        loss=DEFAULT_LOSS,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        extra_pre_call: Optional[PredictionBlock] = None,
        pre_loss: Optional[PredictionBlock] = None,
        **kwargs,
    ) -> "MultiClassClassificationTask":
        pre_call = Softmax(
            schema,
            feature_name,
            bias_initializer=bias_initializer,
            kernel_initializer=kernel_initializer,
        )
        if extra_pre_call:
            pre_call = pre_call.connect(extra_pre_call)

        return cls(
            pre_call=pre_call,
            pre_loss=pre_loss,
            loss=loss,
            **kwargs,
        )

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return inputs

    def metric_results(self, mode: str = None):
        dict_results = {}
        for metric in self.metrics:
            if hasattr(metric, "top_ks"):
                topks = metric.top_ks
                results = metric.result()
                for measure, k in zip(results, topks):
                    dict_results[f"{metric.name}_{k}"] = measure
            else:
                dict_results.update({metric.name: metric.result()})

        return dict_results
