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

from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers import Dense

from merlin.models.tf.losses import LossType, loss_registry
from merlin.models.tf.metrics.ranking import ranking_metrics
from merlin.schema import Schema, Tags

from ...utils.schema import categorical_cardinalities
from ..core import Block, MetricOrMetricClass, PredictionTask
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = "binary_crossentropy"
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
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS,
        **kwargs,
    ):
        output_layer = kwargs.pop("output_layer", None)
        super().__init__(
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )

        self.output_layer = output_layer or tf.keras.layers.Dense(
            1, activation="linear", name=self.child_name("output_layer")
        )
        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        self.output_activation = tf.keras.layers.Activation(
            "sigmoid", dtype="float32", name="prediction"
        )
        self.loss = loss_registry.parse(loss)

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return self.output_activation(self.output_layer(inputs))

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self,
            config,
            {"output_layer": tf.keras.layers.serialize, "loss": tf.keras.losses.serialize},
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)
        config = maybe_deserialize_keras_objects(
            config, ["output_layer"], tf.keras.layers.deserialize
        )

        return super().from_config(config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategFeaturePrediction(Block):
    def __init__(
        self,
        schema: Schema,
        feature_name: str = None,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        activation=None,
        **kwargs,
    ):
        super(CategFeaturePrediction, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.feature_name = feature_name or schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.num_classes = categorical_cardinalities(schema)[self.feature_name]
        self.activation = activation

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        self.output_activation = tf.keras.layers.Activation(
            activation, dtype="float32", name="predictions"
        )

    def build(self, input_shape):
        self.output_layer = Dense(
            units=self.num_classes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.feature_name}-prediction",
            activation="linear",
        )
        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        return self.output_activation(self.output_layer(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_classes,)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MultiClassClassificationTask(PredictionTask):
    DEFAULT_LOSS = "sparse_categ_crossentropy"
    DEFAULT_METRICS = {
        "ranking": ranking_metrics(top_ks=[10, 20]),
        "multi-class": (),
    }

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS["ranking"],
        pre: Optional[Block] = None,
        **kwargs,
    ):

        super().__init__(
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            **kwargs,
        )
        self.loss = loss_registry.parse(loss)

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        feature_name: str = Tags.ITEM_ID,
        loss=DEFAULT_LOSS,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        extra_pre: Optional[Block] = None,
        **kwargs,
    ) -> "MultiClassClassificationTask":
        pre = CategFeaturePrediction(
            schema,
            feature_name,
            bias_initializer=bias_initializer,
            kernel_initializer=kernel_initializer,
        )
        if extra_pre:
            pre = pre.connect(extra_pre)

        return cls(
            pre=pre,
            loss=loss,
            **kwargs,
        )

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        if getattr(self.loss, "sample_weight", None):
            return self.loss(targets, predictions, sample_weight=sample_weight)
        return self.loss(targets, predictions)

    def call(self, inputs, training=False, **kwargs):
        return inputs

    def metric_results(self, mode: str = None):
        dict_results = {}
        for metric in self.metrics:
            if hasattr(metric, "top_ks"):
                topks = metric.top_ks
                results = metric.result()
                for i, k in enumerate(topks):
                    dict_results[f"{metric.name}_{k}"] = results[i]
            else:
                dict_results.update({metric.name: metric.result()})

        return dict_results

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, {"loss": tf.keras.losses.serialize})

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)

        return super().from_config(config)
