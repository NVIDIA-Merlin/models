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
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.losses import LossType, loss_registry

from ..core import PredictionTask
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RegressionTask(PredictionTask):
    DEFAULT_LOSS = "mse"
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        **kwargs,
    ):
        logit = kwargs.pop("logit", None)
        super().__init__(
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )
        self.logit = logit or tf.keras.layers.Dense(1, name=self.child_name("logit"))
        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        self.output_activation = tf.keras.layers.Activation(
            "linear", dtype="float32", name="prediction"
        )
        self.loss = loss_registry.parse(loss)

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        return self.output_activation(self.logit(inputs))

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self, config, {"logit": tf.keras.layers.serialize, "loss": tf.keras.losses.serialize}
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)
        config = maybe_deserialize_keras_objects(config, ["logit"], tf.keras.layers.deserialize)

        return super().from_config(config)
