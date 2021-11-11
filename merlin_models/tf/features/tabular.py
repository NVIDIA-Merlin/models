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

from typing import List, Optional, Tuple, Type, Union, cast

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType
from merlin_standard_lib.utils.doc_utils import docstring_parameter

from ..block.mlp import MLPBlock
from ..core import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    AsTabular,
    Block,
    InputBlockMixin,
    ParallelBlock,
    SequentialBlock,
    TabularAggregationType,
    TabularBlock,
    TabularTransformationType,
)
from ..utils import tf_utils
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures

TABULAR_FEATURES_PARAMS_DOCSTRING = """
    continuous_layer: TabularBlock, optional
        Block used to process continuous features.
    categorical_layer: TabularBlock, optional
        Block used to process categorical features.
    text_embedding_layer: TabularBlock, optional
        Block used to process text features.
"""


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TabularFeatures(ParallelBlock, InputBlockMixin):
    """Input block that combines different types of features: continuous, categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    {tabular_module_parameters}
    """

    CONTINUOUS_MODULE_CLASS: Type[TabularBlock] = ContinuousFeatures
    EMBEDDING_MODULE_CLASS: Type[TabularBlock] = EmbeddingFeatures

    def __init__(
        self,
        continuous_layer: Optional[TabularBlock] = None,
        categorical_layer: Optional[TabularBlock] = None,
        text_embedding_layer: Optional[TabularBlock] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        to_merge = {}
        if continuous_layer:
            to_merge["continuous_layer"] = continuous_layer
        if categorical_layer:
            to_merge["categorical_layer"] = categorical_layer
        if text_embedding_layer:
            to_merge["text_embedding_layer"] = text_embedding_layer

        assert to_merge != [], "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(
            to_merge,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            **kwargs
        )

        if continuous_projection:
            self.project_continuous_features(continuous_projection)

    def project_continuous_features(
        self, block_or_dims: Union[List[int], Block]
    ) -> "TabularFeatures":
        """Combine all concatenated continuous features with stacked MLP layers

        Parameters
        ----------
        mlp_layers_dims : Union[List[int], int]
            The MLP layer dimensions

        Returns
        -------
        TabularFeatures
            Returns the same ``TabularFeatures`` object with the continuous features projected
        """
        if not isinstance(block_or_dims, Block):
            if isinstance(block_or_dims, int):
                block_or_dims = [block_or_dims]

            block = MLPBlock(block_or_dims)
        else:
            block = block_or_dims

        continuous = cast(tf.keras.layers.Layer, self.continuous_layer)
        continuous.set_aggregation("concat")

        continuous = SequentialBlock([continuous, block, AsTabular("continuous_projection")])

        self.parallel_dict["continuous_layer"] = continuous

        return self

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
        aggregation: Optional[str] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        embedding_dim_default: Optional[int] = 64,
        max_sequence_length=None,
        max_text_length=None,
        **kwargs
    ) -> "TabularFeatures":
        maybe_continuous_layer, maybe_categorical_layer = None, None
        if continuous_tags:
            maybe_continuous_layer = cls.CONTINUOUS_MODULE_CLASS.from_schema(
                schema,
                tags=continuous_tags,
            )
        if categorical_tags:
            maybe_categorical_layer = cls.EMBEDDING_MODULE_CLASS.from_schema(
                schema, tags=categorical_tags, embedding_dim_default=embedding_dim_default
            )

        output = cls(
            continuous_layer=maybe_continuous_layer,
            categorical_layer=maybe_categorical_layer,
            aggregation=aggregation,
            continuous_projection=continuous_projection,
            schema=schema,
            **kwargs
        )

        return output

    @property
    def continuous_layer(self) -> Optional[tf.keras.layers.Layer]:
        if "continuous_layer" in self.parallel_dict:
            return self.parallel_dict["continuous_layer"]

        return None

    @property
    def categorical_layer(self) -> Optional[tf.keras.layers.Layer]:
        if "categorical_layer" in self.parallel_dict:
            return self.parallel_dict["categorical_layer"]

        return None

    def get_config(self):
        from merlin_models.tf import TabularBlock as _TabularBlock

        config = tf_utils.maybe_serialize_keras_objects(
            self,
            _TabularBlock.get_config(self),
            ["continuous_layer", "categorical_layer", "text_embedding_layer"],
        )

        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            [
                "continuous_layer",
                "categorical_layer",
                "text_embedding_layer",
                "pre",
                "post",
                "aggregation",
            ],
        )

        if "schema" in config:
            config["schema"] = Schema().from_json(config["schema"])

        return cls(**config)
