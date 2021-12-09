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
import abc
from typing import Optional

import tensorflow as tf
from tensorflow.python.keras.layers import Dot

from merlin_standard_lib import Schema, Tag

from ..api import inputs
from ..core import Block, BlockType, ParallelBlock, TabularAggregation, tabular_aggregation_registry
from ..features.embedding import EmbeddingFeatures, EmbeddingOptions
from ..typing import TabularData


class Distance(TabularAggregation, abc.ABC):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        assert len(inputs) == 2

        return self.distance(inputs, **kwargs)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()


@tabular_aggregation_registry.register("cosine")
class CosineSimilarity(Distance):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dot = Dot(axes=1, normalize=True)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        out = self.dot(list(inputs.values()))

        return out


def TwoTowerBlock(
    schema,
    query_tower: Block,
    item_tower: Optional[Block] = None,
    query_tower_tag=Tag.USER,
    item_tower_tag=Tag.ITEM,
    embedding_dim_default: Optional[int] = 64,
    post: Optional[BlockType] = None,
    # negative_memory_bank=None,
    **kwargs,
) -> ParallelBlock:
    """
    Builds the Two-tower archicture, as proposed in the following
    `paper https://doi.org/10.1145/3298689.3346996`_ [Xinyang19].

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    query_tower : Block
        The `Block` that combines user features
    item_tower : Optional[Block], optional
        The optional `Block` that combines items features
    query_tower_tag : Tag
        The tag to select query features, by default Tag.USER
    item_tower_tag : Tag
        The tag to select item features, by default Tag.ITEM
    embedding_dim_default : Optional[int], optional
        Dimension of the embeddings, by default 64
    post: Optional[Block], optional
        The optional `Block` to apply on both outputs of Two-tower model

    Returns
    -------
    ParallelBlock
        The Two-tower block

    Raises
    ------
    ValueError
        The schema is required by TwoTower
    ValueError
        The query_tower is required by TwoTower
    """
    if schema is None:
        raise ValueError("The schema is required by TwoTower")
    if query_tower is None:
        raise ValueError("The query_tower is required by TwoTower")

    _item_tower: Block = item_tower or query_tower.copy()
    if not getattr(_item_tower, "inputs", None):
        item_schema = schema.select_by_tag(item_tower_tag) if item_tower_tag else schema
        if not item_schema:
            raise ValueError(
                f"The schema should contain features with the tag `{item_tower_tag}`,"
                "required by item-tower"
            )
        _item_tower = inputs(
            item_schema,
            embedding_options=EmbeddingOptions(embedding_dim_default=embedding_dim_default),
        ).connect(_item_tower)
    if not getattr(query_tower, "inputs", None):
        query_schema = schema.select_by_tag(query_tower_tag) if query_tower_tag else schema
        if not query_schema:
            raise ValueError(
                f"The schema should contain features with the tag `{query_schema}`,"
                "required by query-tower"
            )
        query_tower = inputs(
            query_schema,
            embedding_options=EmbeddingOptions(embedding_dim_default=embedding_dim_default),
        ).connect(query_tower)

    two_tower = ParallelBlock({"query": query_tower, "item": _item_tower}, post=post, **kwargs)

    return two_tower


def MatrixFactorizationBlock(
    schema: Schema, dim: int, query_id_tag=Tag.USER_ID, item_id_tag=Tag.ITEM_ID, **kwargs
):
    query_item_schema = schema.select_by_tag(
        lambda tags: query_id_tag in tags or item_id_tag in tags
    )
    matrix_factorization = EmbeddingFeatures.from_schema(
        query_item_schema, options=EmbeddingOptions(embedding_dim_default=dim), **kwargs
    )

    return matrix_factorization
