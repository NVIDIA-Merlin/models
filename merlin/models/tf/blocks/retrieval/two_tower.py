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

from merlin.models.tf.blocks.core.base import Block, BlockType
from merlin.models.tf.blocks.core.inputs import InputBlock
from merlin.models.tf.blocks.core.transformations import L2Norm
from merlin.models.tf.blocks.retrieval.base import DualEncoderBlock, RetrievalMixin
from merlin.models.tf.features.embedding import EmbeddingOptions
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TwoTowerBlock(DualEncoderBlock, RetrievalMixin):
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
    embedding_options : EmbeddingOptions
        Options for the input embeddings.
        - embedding_dims: Optional[Dict[str, int]] - The dimension of the
        embedding table for each feature (key), by default None
        - embedding_dim_default: int - Default dimension of the embedding
        table, when the feature is not found in ``embedding_dims``, by default 64
        - infer_embedding_sizes : bool, Automatically defines the embedding
        dimension from the feature cardinality in the schema, by default False
        - infer_embedding_sizes_multiplier: int. Multiplier used by the heuristic
        to infer the embedding dimension from its cardinality. Generally
        reasonable values range between 2.0 and 10.0. By default 2.0.
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
            query_id_tag=Tags.USER_ID,
            item_id_tag=Tags.ITEM_ID,
            output_ids: bool = True,
            embedding_options: EmbeddingOptions = EmbeddingOptions(
                embedding_dims=None,
                embedding_dim_default=64,
                infer_embedding_sizes=False,
                infer_embedding_sizes_multiplier=2.0,
            ),
            post: Optional[BlockType] = None,
            normalize: bool = True,
            **kwargs,
    ):
        if schema is None:
            raise ValueError("The schema is required by TwoTower")
        if query_tower is None:
            raise ValueError("The query_tower is required by TwoTower")

        _item_tower: Block = _parse_tower(
            item_tower or query_tower .copy(), schema, item_tower_tag, normalize, embedding_options=embedding_options
        )
        query_tower = _parse_tower(
            query_tower, schema, query_tower_tag, normalize, embedding_options=embedding_options
        )

        super().__init__(
            query_block=query_tower,
            item_block=_item_tower,
            query_id_tag=query_id_tag,
            item_id_tag=item_id_tag,
            output_ids=output_ids,
            post=post,
            **kwargs,
        )


def _parse_tower(tower: Block, schema: Schema, tag: str, normalize: bool = True, **kwargs) -> Block:
    if not getattr(tower, "inputs", None):
        schema = schema.select_by_tag(tag)
        if not schema:
            raise ValueError(
                f"The schema should contain features with the tag `{tag}`,"
                "required by the tower"
            )
        tower_inputs = InputBlock(schema)

        tower = tower_inputs.connect(tower)

    if normalize:
        tower = tower.connect(L2Norm())

    return tower
