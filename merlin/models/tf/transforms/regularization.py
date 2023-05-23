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
        """
        Invokes the L2 normalization on the input tensor or dictionary of tensors.

        Parameters
        ----------
        inputs: Union[tf.Tensor, TabularData]
            A Tensor or TabularData input to normalize.
        axis: int, optional
            The axis on which to normalize, by default -1.

        Returns
        -------
        Union[tf.Tensor, TabularData]
            The L2-normalized tensor or dictionary of tensors.
        """
        if isinstance(inputs, dict):
            inputs = {key: self._l2_norm(inp, axis=axis) for key, inp in inputs.items()}
        else:
            inputs = self._l2_norm(inputs, axis=axis)

        return inputs

    def _l2_norm(
        self,
        inputs: Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor],
        epsilon: float = 1e-12,
        axis: int = -1,
    ) -> Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]:
        """Computes L2-norm for a given axis, typically axis = -1.
        Equivalent to tf.linalg.l2_normalize(), but that function
        does not support tf.RaggedTensor

        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
            A dense or sparse/ragged tensor
        epsilon : float, optional
            A small value to add to the sum(vector**2) to avoid div by 0, by default 1e-12
        axis : int, optional
            The axis on which to normalize, by default -1

        Returns
        -------
        Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
            The L2 normalized tensor
        """
        return inputs / tf.math.sqrt(
            tf.math.maximum(tf.reduce_sum(tf.pow(inputs, 2), axis=axis, keepdims=True), epsilon)
        )

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the tensor after normalization.

        Parameters
        ----------
        input_shape : tuple
            A tuple indicating the shape of the input tensor.

        Returns
        -------
        tuple
            The shape of the tensor after L2 normalization.
        """
        return input_shape
