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

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ResidualBlock, SequentialBlock, TabularAggregationType
from merlin.models.tf.core.tabular import Filter, tabular_aggregation_registry
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.models.utils.misc_utils import filter_kwargs
from merlin.schema import Schema, Tags

InitializerType = Union[str, tf.keras.initializers.Initializer]
RegularizerType = Union[str, tf.keras.regularizers.Regularizer]


def MLPBlock(
    dimensions: List[int],
    activation: Union[str, List[str]] = "relu",
    use_bias: bool = True,
    kernel_initializer: InitializerType = "glorot_uniform",
    bias_initializer: InitializerType = "zeros",
    kernel_regularizer: Optional[RegularizerType] = None,
    bias_regularizer: Optional[RegularizerType] = None,
    activity_regularizer: Optional[RegularizerType] = None,
    dropout: Optional[float] = None,
    normalization: Optional[Union[str, tf.keras.layers.Layer]] = None,
    filter: Optional[Union[Schema, Tags, List[str], "Filter"]] = None,
    no_activation_last_layer: bool = False,
    block_name: str = "MLPBlock",
    **kwargs,
) -> SequentialBlock:
    """
    A block that applies a multi-layer perceptron to the input.

    Example usage::
        mlp = ml.InputBlock(schema).connect(ml.MLPBlock([64, 32]))

    Parameters
    ----------
    dimensions: List[int]
        The number of units in each layer of the MLP.
    activation: str
        The activation function to use.
    use_bias: bool
        Whether to use a bias in the MLP.
    kernel_initializer: InitializerType
        Initializer for the kernel weights matrix. Defaults to "glorot_uniform".
    bias_initializer: InitializerType
        Initializer for the bias vector. Default to "zeros".
    kernel_regularizer: Optional[RegularizerType]
        Regularizer function applied to the kernel weights matrix. Default to None.
    bias_regularizer: Optional[RegularizerType]
        Regularizer function applied to the bias vector.  Default to None.
    activity_regularizer: Optional[RegularizerType]
        Regularizer function applied to the output of the layer (its "activation").
        Default to None.
    dropout: float
        The dropout rate to use.
    normalization: str or Layer
        The normalization layer to use.
    filter: Schema, Tag, List[str], or Filter
        The filter to apply to the inputs of the MLP.
    no_activation_last_layer: bool
        Ensures that no activation function (i.e. 'linear') or droptout is used in the
        output of the last MLP layer
    block_name: str
        The name of the block.
    """

    if isinstance(activation, list) and len(activation) != len(dimensions):
        raise ValueError(
            f"Activation and Dimensions length mismatch. \
        Activation length: {len(activation)}, Dimensions length: {len(dimensions)}"
        )

    block_layers = []

    for idx, dim in enumerate(dimensions):
        dropout_layer = None
        activation_idx = activation if isinstance(activation, str) else activation[idx]
        if no_activation_last_layer and idx == len(dimensions) - 1:
            activation_idx = "linear"
        else:
            if dropout:
                if activation_idx in ["selu", tf.keras.activations.selu]:
                    # Best practice for SeLU. It is also recommended
                    # kernel_initializer="lecun_normal"
                    dropout_layer = tf.keras.layers.AlphaDropout(dropout)
                else:
                    dropout_layer = tf.keras.layers.Dropout(dropout)

        block_layers.append(
            _Dense(
                dim,
                activation=activation_idx,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
            )
        )
        if dropout_layer:
            block_layers.append(dropout_layer)

        if normalization:
            if normalization == "batch_norm":
                block_layers.append(tf.keras.layers.BatchNormalization())
            elif isinstance(normalization, tf.keras.layers.Layer):
                block_layers.append(normalization)
            else:
                raise ValueError("Normalization needs to be an instance `Layer` or " "`batch_norm`")

    return SequentialBlock(block_layers, filter=filter, block_name=block_name, **kwargs)


def DenseResidualBlock(
    low_rank_dim: Optional[int] = None,
    activation: Optional[Union[str, tf.keras.layers.Layer]] = "relu",
    use_bias: bool = True,
    dropout: Optional[float] = None,
    normalization: Optional[Union[str, tf.keras.layers.Layer]] = "batch_norm",
    depth: int = 1,
    **dense_kwargs,
) -> Block:
    """A block that applies a dense residual block to the input.
    The residual consists in the input summed element-wise with the
    output of the dense block.
    The dense block projects the inputs using dense layers to the same output dim.
    If the input dim is very high dimensional, the low_rank_dim can
    be used to create a low rank matrix and reduce the necessary number
    of parameters for this projection.

    Example usage::
        block = ml.DenseResidualBlock(depth=3).connect(ml.MLPBlock([1]))

    Parameters
    ----------
    low_rank_dim: int, optional
        The dimension of the low rank matrix. If set, it projects the input to the
        low_rank_dim (`LR`) and then back to the input dim (`I`) as output.
        That requires much less parameters (`I*LR + LR*I`) than projecting the
        inputs dim directly to the same dim (`I*I`). By default None
    activation: Union[str, tf.keras.layers.Layer], optional
        The activation function to use. By default "relu"
    use_bias: bool
        Whether to use a bias in the MLP. By default True
    dropout: float
        The dropout rate to use. By default 0.0
    normalization: Union[str, tf.keras.layers.Layer], optional
        The normalization layer to use. By the default None.
    depth: int
        The number of residual blocks to stack. By default 1
    """

    block_layers = []
    block_layers.append(
        DenseMaybeLowRank(low_rank_dim, activation=None, use_bias=use_bias, **dense_kwargs)
    )
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
    elif depth < 1:
        raise ValueError(
            "The depth (number of stacked residual blocks) needs " "to be equal or greater than 1."
        )

    return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class _Dense(tf.keras.layers.Layer):
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
        **kwargs,
    ):
        super(_Dense, self).__init__(**kwargs)
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
            **kwargs,
        )
        self.pre_aggregation = pre_aggregation
        self.units = units

    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            inputs = tabular_aggregation_registry.parse(self.pre_aggregation)(inputs)

        filtered_kwargs = filter_kwargs(kwargs, self.dense)
        return self.dense(inputs, **filtered_kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            agg = tabular_aggregation_registry.parse(self.pre_aggregation)
            input_shape = agg.compute_output_shape(input_shape)

        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        config = super(_Dense, self).get_config()
        config["pre_aggregation"] = self.pre_aggregation
        config["units"] = self.units

        return maybe_serialize_keras_objects(self, config, ["dense"])

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, {"dense": tf.keras.layers.deserialize})

        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DenseMaybeLowRank(tf.keras.layers.Layer):
    def __init__(
        self,
        low_rank_dim: Optional[int] = None,
        use_bias: bool = True,
        activation: Optional[Union[str, tf.keras.layers.Layer]] = None,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        pre_aggregation: Optional[TabularAggregationType] = "concat",
        dense: Optional[tf.keras.layers.Dense] = None,
        dense_u: Optional[tf.keras.layers.Dense] = None,
        **kwargs,
    ):
        """A block that projects the inputs the same input dim.
        If the input dim is very high dimensional, the low_rank_dim can
        be used to create a low rank matrix and reduce the necessary number
        of parameters for this projection.

        Parameters
        ----------
        low_rank_dim : Optional[int], optional
            The dimension of the low rank matrix. If set, it projects the input to the
            low_rank_dim (`LR`) and then back to the input dim (`I`) as output.
            That requires much less parameters (`I*LR + LR*I`) than projecting the
            inputs dim directly to the same dim (`I*I`), by default None.
        use_bias : bool, optional
            Whether to use a bias in the MLP, by default True
        activation : Optional[Union[str,tf.keras.layers.Layer]], optional
            The activation function to use. By default None
        kernel_initializer: InitializerType
            Initializer for the kernel weights matrix. Defaults to "glorot_uniform".
        bias_initializer: InitializerType
            Initializer for the bias vector. Default to "zeros".
        kernel_regularizer: Optional[RegularizerType]
            Regularizer function applied to the kernel weights matrix. Default to None.
        bias_regularizer: Optional[RegularizerType]
            Regularizer function applied to the bias vector.  Default to None.
        pre_aggregation : Optional[TabularAggregationType], optional
            Aggregation to be done before projection, by default "concat"
        dense : Optional[tf.keras.layers.Dense], optional
            An optional dense layer to be used for projection,
            by default None. If not set it is created internally.
        dense_u : Optional[tf.keras.layers.Dense], optional
            An optional dense layer to be called first if low_rank_dim is set,
            by default None. If not set it is created internally.
        """

        super().__init__(**kwargs)
        self.low_rank_dim = low_rank_dim
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.pre_aggregation = pre_aggregation
        self.dense = dense
        self.dense_u = dense_u

    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self.dense is None:
            self.dense = _Dense(
                last_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
            )

        if self.low_rank_dim is not None and self.dense_u is None:
            self.dense_u = _Dense(
                self.low_rank_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
            )
        super(DenseMaybeLowRank, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        if isinstance(inputs, dict):
            inputs = tabular_aggregation_registry.parse(self.pre_aggregation)(inputs)

        if self.low_rank_dim is None:
            return self.dense(inputs)  # type: ignore

        return self.dense(self.dense_u(inputs))  # type: ignore

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            agg = tabular_aggregation_registry.parse(self.pre_aggregation)
            input_shape = agg.compute_output_shape(input_shape)

        return input_shape

    def get_config(self):
        config = dict(
            low_rank_dim=self.low_rank_dim,
            use_bias=self.use_bias,
            activation=self.activation,
            pre_aggregation=self.pre_aggregation,
        )
        config.update(super(DenseMaybeLowRank, self).get_config())

        config = maybe_serialize_keras_objects(
            self,
            config,
            [
                "dense",
                "dense_u",
            ],
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(
            config,
            [
                "dense",
                "dense_u",
            ],
        )

        return cls(**config)
