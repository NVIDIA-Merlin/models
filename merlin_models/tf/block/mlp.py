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

from typing import List, Optional, Union

import tensorflow as tf

from merlin_standard_lib import Schema, Tag

from ..core import Block, Filter, ResidualBlock, SequentialBlock, tabular_aggregation_registry
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects


def MLPBlock(
    dimensions: List[int],
    activation="relu",
    use_bias: bool = True,
    dropout: Optional[float] = None,
    normalization: Optional[Union[str, tf.keras.layers.Layer]] = None,
    filter: Optional[Union[Schema, Tag, List[str], "Filter"]] = None,
    block_name: str = "MLPBlock",
    **kwargs
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

    return SequentialBlock(block_layers, filter=filter, block_name=block_name, **kwargs)


def DenseResidualBlock(
    projection_dim: Optional[int] = None,
    activation="relu",
    use_bias: bool = True,
    dropout: Optional[float] = None,
    normalization: Optional[Union[str, tf.keras.layers.Layer]] = "batch_norm",
    depth: int = 1,
) -> Block:
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

    output = ResidualBlock(
        SequentialBlock(block_layers, block_name="DenseResidual"), activation=activation
    )

    if depth > 1:
        return output.repeat(depth - 1)

    return output


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


InitializerType = Union[str, tf.keras.initializers.Initializer]
RegularizerType = Union[str, tf.keras.regularizers.Regularizer]


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DenseSameDim(tf.keras.layers.Layer):
    def __init__(
        self,
        projection_dim: Optional[int] = None,
        use_bias: bool = True,
        activation=None,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        pre_aggregation="concat",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.pre_aggregation = pre_aggregation

    def build(self, input_shape):
        last_dim = input_shape[-1]

        dense = tf.keras.layers.Dense(
            last_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
        )

        if self.projection_dim is None:
            self.dense = dense
        else:
            self.dense_u = tf.keras.layers.Dense(
                self.projection_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
            )
            self.dense_v = dense
        super(DenseSameDim, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        if isinstance(inputs, dict):
            inputs = tabular_aggregation_registry.parse(self.pre_aggregation)(inputs)

        if self.projection_dim is None:
            return self.dense(inputs)

        return self.dense_v(self.dense_u(inputs))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            agg = tabular_aggregation_registry.parse(self.pre_aggregation)
            input_shape = agg.compute_output_shape(input_shape)

        return input_shape

    def get_config(self):
        config = dict(
            projection_dim=self.projection_dim,
            use_bias=self.use_bias,
            activation=self.activation,
            pre_aggregation=self.pre_aggregation,
        )
        config.update(super(DenseSameDim, self).get_config())

        return maybe_serialize_keras_objects(
            self,
            config,
            ["kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer"],
        )
