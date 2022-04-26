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
from tensorflow.python.keras.layers import Dense

from merlin.models.tf.blocks.core.base import Block, MetricOrMetrics
from merlin.models.tf.losses import LossType, loss_registry
from merlin.models.tf.prediction_tasks.base import PredictionTask
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.models.utils.schema_utils import categorical_cardinalities
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalPrediction(Block):
    """Block that predicts a categorical feature. num_classes is inferred from the"""

    def __init__(
        self,
        schema: Schema,
        feature_name: Optional[str] = None,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        activation=None,
        **kwargs,
    ):
        super(CategoricalPrediction, self).__init__(**kwargs)
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
    """
    Prediction task for multi-class classification.

    Parameters
    ----------
    target_name : Optional[str], optional
        Label name, by default None
    task_name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    loss: LossType, optional
        The loss to use for the task.
        Defaults to "sparse_categorical_crossentropy".
    metrics: MetricOrMetrics, optional
        The metrics to use for the task. Defaults to [accuracy].
    """

    DEFAULT_LOSS = "sparse_categorical_crossentropy"
    SUPPORTED_LOSSES = [
        "sparse_categorical_crossentropy",
        "categorical_crossentropy"
    ]
    DEFAULT_METRICS: MetricOrMetrics = (tf.keras.metrics.Accuracy,)

    def __init__(
        self,
        block: Block,
        target_name: Optional[Union[str, Schema]] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        **kwargs,
    ):
        super().__init__(
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            post=post,
            **kwargs,
        )
        self.block = block

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        feature_name: str = Tags.ITEM_ID,
        loss=DEFAULT_LOSS,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        **kwargs,
    ) -> "MultiClassClassificationTask":
        """Create from Schema."""
        block = CategoricalPrediction(
            schema,
            feature_name,
            bias_initializer=bias_initializer,
            kernel_initializer=kernel_initializer,
        )

        return cls(
            block,
            pre=pre,
            post=post,
            loss=loss,
            **kwargs,
        )

    def call(self, inputs, training=False, **kwargs):
        return self.block(inputs, training=training, **kwargs)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, {"loss": tf.keras.losses.serialize})

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)

        return super().from_config(config)
