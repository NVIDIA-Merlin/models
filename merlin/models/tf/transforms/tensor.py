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
from typing import Dict, Union

import tensorflow as tf
from keras.layers.preprocessing import preprocessing_utils as utils

from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.typing import TabularData

ONE_HOT = utils.ONE_HOT
MULTI_HOT = utils.MULTI_HOT
COUNT = utils.COUNT


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExpandDims(TabularBlock):
    """
    Expand dims of selected input tensors.
    Example::
        inputs = {
            "cont_feat1": tf.random.uniform((NUM_ROWS,)),
            "cont_feat2": tf.random.uniform((NUM_ROWS,)),
            "multi_hot_categ_feat": tf.random.uniform(
                (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
            ),
        }
        expand_dims_op = tr.ExpandDims(expand_dims={"cont_feat2": 0, "multi_hot_categ_feat": 1})
        expanded_inputs = expand_dims_op(inputs)
    """

    def __init__(self, expand_dims: Union[int, Dict[str, int]] = -1, **kwargs):
        """Instantiates the `ExpandDims` transformation, which allows to expand dims
        of the input tensors
        Parameters
        ----------
        expand_dims : Union[int, Dict[str, int]], optional, by default -1
            Defines which dimensions should be expanded. If an `int` is provided, all input tensors
            will have the same dimension expanded. If a `dict` is passed, only features matching
            the dict keys will be expanded, in the dimension specified as the dict values.
        """
        super().__init__(**kwargs)
        self.inputs_expand_dims = expand_dims

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}

        for k, v in inputs.items():
            if isinstance(self.inputs_expand_dims, int):
                outputs[k] = tf.expand_dims(v, self.inputs_expand_dims)
            elif isinstance(self.inputs_expand_dims, dict) and k in self.inputs_expand_dims:
                expand_dim = self.inputs_expand_dims[k]
                outputs[k] = tf.expand_dims(v, expand_dim)
            elif self.inputs_expand_dims:
                outputs[k] = v
            else:
                raise ValueError("The expand_dims argument is not valid")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def to_dense(tensor, max_seq_length=None):
    if isinstance(tensor, tf.RaggedTensor):
        if max_seq_length:
            shape = [None] * tensor.shape.rank
            shape[1] = max_seq_length
            return tensor.to_tensor(shape=shape)
        result = tensor.to_tensor()

    elif isinstance(tensor, tf.SparseTensor):
        result = tf.sparse.to_dense(tensor)

    elif isinstance(tensor, tf.Tensor):
        result = tensor

    else:
        result = tf.convert_to_tensor(tensor)

    return result


def to_sparse(tensor):
    if isinstance(tensor, tf.RaggedTensor):
        result = tensor.to_sparse()
    elif isinstance(tensor, tf.Tensor):
        result = tf.sparse.from_dense(tensor)
    elif isinstance(tensor, tf.SparseTensor):
        result = tensor
    else:
        raise ValueError(
            "Only tf.RaggedTensor and tf.Tensor are acceptable "
            f"for converting to tf.SparseTensor, but got a {type(tensor)}"
        )
    return result
