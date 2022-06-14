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
from typing import Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.prediction_tasks.base import PredictionTask
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RegressionTask(PredictionTask):
    """
    Prediction task for regression-task.

    Parameters
    ----------
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    task_name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    metrics: MetricOrMetrics, optional
        The metrics to use for the task. Defaults to [root-mean-squared-error].
    """

    DEFAULT_LOSS = "mse"
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target: Optional[Union[str, Schema]] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        **kwargs,
    ):
        if isinstance(target, Schema):
            target_name = target.select_by_tag(Tags.REGRESSION)
            if not target_name.column_names:
                raise ValueError(
                    "Regression task requires a column with a ", "`Tags.REGRESSION` tag."
                )
            elif len(target_name.column_names) > 1:
                raise ValueError(
                    "Regression task requires a single column with a ",
                    "`Tags.REGRESSION` tag.",
                    "Found {} columns. ".format(len(target_name.column_names)),
                    "Please specify the column name with the `target` argument.",
                )
            target_name = target_name.column_names[0]
        else:
            target_name = target if target else kwargs.pop("target_name", None)

        logit = kwargs.pop("logit", None)
        super().__init__(
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

    def call(self, inputs, training=False, **kwargs):
        return self.output_activation(self.logit(inputs))

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, {"logit": tf.keras.layers.serialize})

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["logit"], tf.keras.layers.deserialize)

        return super().from_config(config)
