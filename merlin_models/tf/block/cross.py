from typing import List, Optional, Union

import tensorflow as tf

from merlin_standard_lib import Schema, Tag

from ..core import Filter, SequentialBlock, is_input_block
from ..utils.tf_utils import maybe_serialize_keras_objects
from .mlp import DenseSameDim, InitializerType, RegularizerType


def CrossBlock(
    depth: int = 1,
    filter: Optional[Union[Schema, Tag, List[str], Filter]] = None,
    projection_dim: Optional[int] = None,
    diagonal_scale: Optional[float] = 0.0,
    use_bias: bool = True,
    kernel_initializer: InitializerType = "truncated_normal",
    bias_initializer: InitializerType = "zeros",
    kernel_regularizer: Optional[RegularizerType] = None,
    bias_regularizer: Optional[RegularizerType] = None,
    inputs: Optional[tf.keras.layers.Layer] = None,
    **kwargs
):
    if inputs and is_input_block(inputs) and not inputs.aggregation:
        inputs.set_aggregation("concat")

    layers = [inputs] if inputs else []

    for i in range(depth):
        layers.append(
            Cross(
                projection_dim=projection_dim,
                diagonal_scale=diagonal_scale,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                output_x0=i < depth - 1,
            )
        )

    return SequentialBlock(layers, filter=filter, block_name="CrossBlock", **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class Cross(tf.keras.layers.Layer):
    def __init__(
        self,
        projection_dim: Optional[int] = None,
        diagonal_scale: Optional[float] = 0.0,
        use_bias: bool = True,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        output_x0: bool = False,
        **kwargs
    ):
        super(Cross, self).__init__(**kwargs)

        self.diagonal_scale = diagonal_scale
        self.output_x0 = output_x0
        self.dense = kwargs.get(
            "dense",
            DenseSameDim(
                projection_dim=projection_dim,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            ),
        )

        self._supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: tf.Tensor, **kwargs):
        if isinstance(inputs, tf.Tensor):
            x, x0 = inputs, inputs
        else:
            x0, x = inputs

        self.validate_inputs(x0, x)

        projected = self.dense(x)

        if self.diagonal_scale:
            projected = projected + self.diagonal_scale * x

        output = x0 * projected + x
        if self.output_x0:
            return x0, output

        return output

    def validate_inputs(self, x0, x):
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError(
                "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
                "dimension {}. This case is not supported yet.".format(x0.shape[-1], x.shape[-1])
            )

    def get_config(self):
        config = dict(diagonal_scale=self.diagonal_scale)
        config.update(super(Cross, self).get_config())

        return maybe_serialize_keras_objects(self, config, ["dense"])
