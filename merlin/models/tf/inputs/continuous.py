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

from typing import List, Optional, Sequence, Union, overload

import tensorflow as tf

from merlin.models.tf.blocks.core.base import BlockType
from merlin.models.tf.blocks.core.tabular import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Schema, Tags


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContinuousFeatures(TabularBlock):
    """Input block for continuous features.

    Parameters
    ----------
    features: List[str]
        List of continuous features to include in this module.
    {tabular_module_parameters}
    """

    @overload
    def __init__(
        self,
        inputs: Sequence[str],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        max_seq_length: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        ...

    @overload
    def __init__(
        self,
        inputs: Union[Schema, Tags],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        max_seq_length: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        ...

    def __init__(
        self,
        inputs,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        max_seq_length: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        if isinstance(inputs, Schema):
            if not inputs.column_schemas:
                raise ValueError("Schema must contain at least one column")
            max_seq_length = schema_utils.max_value_count(inputs)

        if max_seq_length:
            as_dense = AsDenseFeatures(max_seq_length)
            pre = as_dense if pre is None else [pre, as_dense]

        super().__init__(
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            is_input=True,
            **kwargs
        )
        self.filter_features = Filter(inputs, schema=schema)

    def call(self, inputs, *args, **kwargs):
        cont_features = self.filter_features(inputs)
        cont_features = {
            k: tf.expand_dims(v, -1) if len(v.shape) == 1 else v for k, v in cont_features.items()
        }
        return cont_features

    def compute_call_output_shape(self, input_shapes):
        cont_features_sizes = self.filter_features.compute_output_shape(input_shapes)
        cont_features_sizes = {
            k: tf.TensorShape(list(v) + [1]) if len(v) == 1 else v
            for k, v in cont_features_sizes.items()
        }
        return cont_features_sizes

    def get_config(self):
        config = super().get_config()

        config["inputs"] = self.filter_features.feature_names

        return config

    def _get_name(self):
        return "ContinuousFeatures"

    def repr_ignore(self) -> List[str]:
        return ["filter_features"]

    def repr_extra(self):
        return ", ".join(sorted(self.filter_features.feature_names))
