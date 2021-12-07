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

from merlin_standard_lib import Schema, Tag

from ..core import Block, ParallelBlock, TabularTransformationsType, merge
from ..features.embedding import EmbeddingFeatures
from .inputs import TabularFeatures


def TwoTowerBlock(
    schema,
    query_tower: Block,
    item_tower: Optional[Block] = None,
    query_tower_tag=Tag.USER,
    item_tower_tag=Tag.ITEM,
    add_to_query_context: List[Union[str, Tag]] = None,
    add_to_item_context: List[Union[str, Tag]] = None,
    embedding_dim_default: Optional[int] = 64,
    post: Optional[TabularTransformationsType] = None,
    **kwargs
) -> ParallelBlock:
    _item_tower: Block = item_tower or query_tower.copy()
    if not getattr(_item_tower, "inputs", None):
        item_schema = schema.select_by_tag(item_tower_tag) if item_tower_tag else schema
        _item_tower = TabularFeatures(
            item_schema,
            embedding_dim_default=embedding_dim_default,
            add_to_context=add_to_item_context,
        ).connect(_item_tower)
    if not getattr(query_tower, "inputs", None):
        query_schema = schema.select_by_tag(query_tower_tag) if query_tower_tag else schema
        query_tower = TabularFeatures(
            query_schema,
            embedding_dim_default=embedding_dim_default,
            add_to_context=add_to_query_context,
        ).connect(query_tower)

    two_tower = ParallelBlock({"query": query_tower, "item": _item_tower}, post=post, **kwargs)

    return two_tower


def MatrixFactorizationBlock(
    schema: Schema,
    dim: int,
    query_id_tag=Tag.USER_ID,
    item_id_tag=Tag.ITEM_ID,
    distance="cosine",
    **kwargs
):
    query_id, item_id = schema.select_by_tag(query_id_tag), schema.select_by_tag(item_id_tag)
    matrix_factorization = merge(
        {
            str(query_id_tag): EmbeddingFeatures.from_schema(query_id, embedding_dim_default=dim),
            str(item_id_tag): EmbeddingFeatures.from_schema(item_id, embedding_dim_default=dim),
        },
        aggregation=distance,
        **kwargs
    )

    return matrix_factorization
