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

from typing import Optional

from merlin_standard_lib import Schema, Tag

from ..core import Block, Filter, ParallelBlock, SequentialBlock, TabularBlock, inputs
from ..layers import DotProductInteraction
from ..tabular.transformations import ExpandDims
from .inputs import ContinuousEmbedding


def DLRMBlock(
    schema: Schema,
    bottom_block: Block,
    top_block: Optional[Block] = None,
    embedding_dim: Optional[int] = None,
) -> SequentialBlock:
    if schema is None:
        raise ValueError("The schema is required by DLRM")
    if bottom_block is None:
        raise ValueError("The bottom_block is required by DLRM")
    if embedding_dim is not None and embedding_dim != bottom_block.layers[-1].units:
        raise ValueError(
            f"The embedding_dim ({embedding_dim}) needs to match the "
            "last layer of bottom MLP ({bottom_block.layers[-1].units})"
        )

    embedding_dim = embedding_dim or bottom_block.layers[-1].units

    embeddings_by_type = {}
    categ_features_schema = schema.select_by_tag(Tag.CATEGORICAL)
    if len(categ_features_schema) > 0:
        categ_embeddings = inputs(
            categ_features_schema, embedding_dim_default=embedding_dim, aggregation="stack"
        )
        embeddings_by_type["categorical"] = categ_embeddings

    continuous_features_schema = schema.select_by_tag(Tag.CONTINUOUS)
    if len(continuous_features_schema) > 0:

        continuous_embedding = ContinuousEmbedding(
            inputs(continuous_features_schema), embedding_block=bottom_block
        )
        embeddings_by_type["continuous"] = continuous_embedding

    embeddings_by_type_block = ParallelBlock(embeddings_by_type)

    fm_interaction_layer = TabularBlock(
        pre=ExpandDims(expand_dims={"continuous": -1}), aggregation="concat"
    ).connect(DotProductInteraction())

    dlrm = embeddings_by_type_block.connect_with_shortcut(
        fm_interaction_layer, shortcut_filter=Filter("continuous"), aggregation="concat"
    )

    if top_block:
        dlrm = dlrm.connect(top_block)

    return dlrm
