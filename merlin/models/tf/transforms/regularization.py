from typing import Union

import tensorflow as tf

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.typing import TabularData


@Block.registry.register_with_multiple_names("l2-norm")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class L2Norm(TabularBlock):
    """Apply L2-normalization to input tensors along a given axis"""

    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)

    def call(self, inputs: Union[tf.Tensor, TabularData], axis: int = -1, **kwargs):
        if isinstance(inputs, dict):
            inputs = {key: tf.linalg.l2_normalize(inp, axis=axis) for key, inp in inputs.items()}
        else:
            inputs = tf.linalg.l2_normalize(inputs, axis=axis)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape
