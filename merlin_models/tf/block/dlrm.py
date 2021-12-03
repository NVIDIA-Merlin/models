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

from ..core import Block, Filter, SequentialBlock, TabularBlock, merge
from ..features.continuous import ContinuousFeatures
from ..features.embedding import EmbeddingFeatures
from ..layers import DotProductInteraction


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

    con_schema, cat_schema = schema.select_by_tag(Tag.CONTINUOUS), schema.select_by_tag(
        Tag.CATEGORICAL
    )

    inputs = {}
    if len(con_schema) > 0:
        inputs["continuous"] = ContinuousFeatures.from_schema(con_schema).connect(bottom_block)

    if len(cat_schema) > 0:
        embedding_dim = embedding_dim or bottom_block.layers[-1].units
        inputs["categorical"] = EmbeddingFeatures.from_schema(
            cat_schema, embedding_dim_default=embedding_dim
        )

    if not top_block:
        return merge(inputs, aggregation="stack").connect(DotProductInteraction())

    dot_product = TabularBlock(aggregation="stack").connect(DotProductInteraction())
    top_block_inputs = (
        merge(inputs)
        .connect_with_shortcut(
            dot_product, shortcut_filter=Filter("continuous"), aggregation="concat"
        )
        .connect(top_block)
    )
    return top_block_inputs
