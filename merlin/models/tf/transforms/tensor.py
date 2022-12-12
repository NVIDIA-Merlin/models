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
from typing import Dict, Optional, Union

import tensorflow as tf
from keras.layers.preprocessing import preprocessing_utils as utils

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import list_col_to_ragged
from merlin.models.utils.schema_utils import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema

ONE_HOT = utils.ONE_HOT
MULTI_HOT = utils.MULTI_HOT
COUNT = utils.COUNT


@Block.registry.register("list-to-ragged")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ListToRagged(TabularBlock):
    """Convert all list (multi-hot/sequential) features to tf.RaggedTensor"""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                outputs[name] = list_col_to_ragged(val)
            elif isinstance(val, tf.SparseTensor):
                outputs[name] = tf.RaggedTensor.from_sparse(val)
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shapes):
        output_shapes = {}
        for k, v in input_shapes.items():
            # If it is a list/sparse feature (in tuple representation), uses the offset as shape
            if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                output_shapes[k] = tf.TensorShape([v[1][0], None])
            else:
                output_shapes[k] = v

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ProcessList(TabularBlock):
    """Process all list (multi-hot/sequential) features.add()

    In NVTabular, list-columns are represented as a tuple of (values, offsets).
    This layer processes those columns and:
    - Converts them to a `tf.RaggedTensor` if the features has a variable length.
    - Converts them to a `tf.Tensor` if the feature has a fixed length.
    """

    def __init__(self, schema: Schema, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}

        for name, val in inputs.items():
            is_ragged = True
            if name in self.schema:
                val_count = self.schema[name].properties.get("value_count")
                if val_count and val_count["min"] == val_count["max"]:
                    is_ragged = False

            if isinstance(val, tuple):
                ragged = list_col_to_ragged(val)
            elif isinstance(val, tf.SparseTensor):
                ragged = tf.RaggedTensor.from_sparse(val)
            else:
                outputs[name] = val
                continue

            if is_ragged:
                if len(ragged.shape) == 2:
                    ragged = tf.expand_dims(ragged, axis=-1)

                outputs[name] = ragged
            else:
                outputs[name] = _ragged_to_dense(ragged)

        return outputs

    def compute_output_shape(self, input_shapes):
        output_shapes = {}
        for k, v in input_shapes.items():
            # If it is a list/sparse feature (in tuple representation), uses the offset as shape
            if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                is_ragged = True
                max_seq_length = None
                if k in self.schema:
                    val_count = self.schema[k].properties.get("value_count")
                    if val_count and val_count["min"] == val_count["max"]:
                        is_ragged = False
                        max_seq_length = val_count["min"]

                if is_ragged:
                    output_shapes[k] = tf.TensorShape([v[1][0], None, 1])
                else:
                    output_shapes[k] = tf.TensorShape([v[1][0], max_seq_length])
            else:
                output_shapes[k] = v

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    def get_config(self):
        config = super().get_config()

        config["schema"] = schema_to_tensorflow_metadata_json(self.schema)

        return config

    @classmethod
    def from_config(cls, config):
        schema = tensorflow_metadata_json_to_schema(config.pop("schema"))

        return cls(schema, **config)


@Block.registry.register("list-to-sparse")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ListToSparse(TabularBlock):
    """Convert all list-inputs to sparse-tensors.

    By default, the dataloader will represent list-columns as a tuple of values & row-lengths.

    """

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                val = list_col_to_ragged(val)
            if isinstance(val, tf.RaggedTensor):
                outputs[name] = val.to_sparse()
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@Block.registry.register("list-to-dense")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ListToDense(TabularBlock):
    """Convert all list-inputs to dense-tensors.

    By default, the dataloader will represent list-columns as a tuple of values & row-lengths.


    Parameters
    ----------
    max_seq_length : int
        The maximum length of multi-hot features.
    """

    def __init__(self, max_seq_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length

    def call(self, inputs: Union[tuple, tf.RaggedTensor, TabularData], **kwargs) -> TabularData:
        if isinstance(inputs, (tf.SparseTensor, tf.RaggedTensor, tuple)):
            return self._convert_tensor_to_dense(inputs)

        outputs = {}
        for name, val in inputs.items():
            outputs[name] = self._convert_tensor_to_dense(val)
        return outputs

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, dict):
            return self._get_output_tensor_shape(input_shape)

        outputs = {}
        for key, val in input_shape.items():
            outputs[key] = self._get_output_tensor_shape(val)
        return outputs

    def _get_output_tensor_shape(self, val_shape):
        if isinstance(val_shape, tuple) and isinstance(val_shape[1], tf.TensorShape):
            val_shape = val_shape[1]
            return tf.TensorShape([val_shape[0], self.max_seq_length])
        if val_shape.rank > 1 and val_shape[-1] != 1:
            shapes = val_shape.as_list()
            if self.max_seq_length:
                shapes[1] = self.max_seq_length
            return tf.TensorShape(shapes)
        return tf.TensorShape((val_shape[0]))

    def _convert_tensor_to_dense(self, val):
        if isinstance(val, tuple):
            val = list_col_to_ragged(val)
        if isinstance(val, tf.SparseTensor):
            val = tf.RaggedTensor.from_sparse(val)
        if isinstance(val, tf.RaggedTensor):
            return _ragged_to_dense(val, self.max_seq_length)
        return tf.squeeze(val)

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_length": self.max_seq_length})

        return config


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


def _ragged_to_dense(ragged_tensor, max_seq_length=None):
    if max_seq_length:
        shape = [None] * ragged_tensor.shape.rank
        shape[1] = max_seq_length
        return ragged_tensor.to_tensor(shape=shape)
    return tf.squeeze(ragged_tensor.to_tensor())
