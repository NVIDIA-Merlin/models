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

import logging
from typing import Optional

import tensorflow as tf

from merlin.models.tf.blocks.base import Block, BlockType, ParallelBlock
from merlin.models.tf.blocks.core.inputs import InputBlock
from merlin.models.tf.blocks.retrieval.base import RetrievalMixin, TowerBlock
from merlin.models.tf.features.embedding import EmbeddingOptions
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TwoTowerBlock(ParallelBlock, RetrievalMixin):
    """
    Builds the Two-tower architecture, as proposed in the following
    `paper https://doi.org/10.1145/3298689.3346996`_ [Xinyang19].

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    query_tower : Block
        The `Block` that combines user features
    item_tower : Optional[Block], optional
        The optional `Block` that combines items features.
        If not provided, a copy of the query_tower is used.
    query_tower_tag : Tag
        The tag to select query features, by default `Tags.USER`
    item_tower_tag : Tag
        The tag to select item features, by default `Tags.ITEM`
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

    def __init__(
        self,
        schema: Schema,
        query_tower: Block,
        item_tower: Optional[Block] = None,
        query_tower_tag=Tags.USER,
        item_tower_tag=Tags.ITEM,
        embedding_dim_default: Optional[int] = 64,
        post: Optional[BlockType] = None,
        **kwargs,
    ):
        if schema is None:
            raise ValueError("The schema is required by TwoTower")
        if query_tower is None:
            raise ValueError("The query_tower is required by TwoTower")

        _item_tower: Block = item_tower or query_tower.copy()
        embedding_options = EmbeddingOptions(embedding_dim_default=embedding_dim_default)
        if not getattr(_item_tower, "inputs", None):
            item_schema = schema.select_by_tag(item_tower_tag) if item_tower_tag else schema
            if not item_schema:
                raise ValueError(
                    f"The schema should contain features with the tag `{item_tower_tag}`,"
                    "required by item-tower"
                )
            item_tower_inputs = InputBlock(item_schema, embedding_options=embedding_options)
            _item_tower = item_tower_inputs.connect(_item_tower)
        if not getattr(query_tower, "inputs", None):
            query_schema = schema.select_by_tag(query_tower_tag) if query_tower_tag else schema
            if not query_schema:
                raise ValueError(
                    f"The schema should contain features with the tag `{query_schema}`,"
                    "required by query-tower"
                )
            query_inputs = InputBlock(query_schema, embedding_options=embedding_options)
            query_tower = query_inputs.connect(query_tower)
        branches = {"query": TowerBlock(query_tower), "item": TowerBlock(_item_tower)}

        super().__init__(branches, post=post, **kwargs)

    def query_block(self) -> TowerBlock:
        query_tower = self["query"]

        return query_tower

    def item_block(self) -> TowerBlock:
        item_tower = self["item"]

        return item_tower

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)
        output = ParallelBlock(inputs, **config)
        output.__class__ = cls

        return output
