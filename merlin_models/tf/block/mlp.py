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

from typing import List, Optional

import tensorflow as tf

from ..core import ResidualBlock, SequentialBlock, tabular_aggregation_registry
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects
from .cross import DenseSameDim


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class Dense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        pre_aggregation="concat",
        dense=None,
        **kwargs
    ):
        super(Dense, self).__init__(**kwargs)
        self.dense = dense or tf.keras.layers.Dense(
            units,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs
        )
        self.pre_aggregation = pre_aggregation
        self.units = units

    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            inputs = tabular_aggregation_registry.parse(self.pre_aggregation)(inputs)

        return self.dense(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            agg = tabular_aggregation_registry.parse(self.pre_aggregation)
            input_shape = agg.compute_output_shape(input_shape)

        return super(Dense, self).compute_output_shape(input_shape)

    def get_config(self):
        config = super(Dense, self).get_config()
        config["pre_aggregation"] = self.pre_aggregation
        config["units"] = self.units

        return maybe_serialize_keras_objects(self, config, ["dense"])

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, {"dense": tf.keras.layers.deserialize})

        return cls(**config)


def MLPBlock(
    dimensions: List[int],
    activation="relu",
    use_bias: bool = True,
    dropout=None,
    normalization=None,
) -> SequentialBlock:
    block_layers = []

    for dim in dimensions:
        block_layers.append(Dense(dim, activation=activation, use_bias=use_bias))
        if dropout:
            block_layers.append(tf.keras.layers.Dropout(dropout))
        if normalization:
            if normalization == "batch_norm":
                block_layers.append(tf.keras.layers.BatchNormalization())
            elif isinstance(normalization, tf.keras.layers.Layer):
                block_layers.append(normalization)
            else:
                raise ValueError("Normalization needs to be an instance `Layer` or " "`batch_norm`")

    return SequentialBlock(block_layers, block_name="MLPBlock")


def DenseResidualBlock(
    projection_dim: Optional[int] = None,
    activation="relu",
    use_bias: bool = True,
    dropout=None,
    normalization="batch_norm",
) -> ResidualBlock:
    block_layers = []
    block_layers.append(DenseSameDim(projection_dim, activation=None, use_bias=use_bias))
    if dropout:
        block_layers.append(tf.keras.layers.Dropout(dropout))
    if normalization:
        if normalization == "batch_norm":
            block_layers.append(tf.keras.layers.BatchNormalization())
        elif isinstance(normalization, tf.keras.layers.Layer):
            block_layers.append(normalization)
        else:
            raise ValueError("Normalization needs to be an instance `Layer` or " "`batch_norm`")

    return ResidualBlock(
        SequentialBlock(block_layers, block_name="DenseResidual"), activation=activation
    )
