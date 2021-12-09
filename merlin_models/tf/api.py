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


from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType

from .core import (
    Block,
    BlockType,
    Filter,
    ParallelBlock,
    ParallelPredictionBlock,
    SequentialBlock,
    TabularAggregationType,
    TabularBlock,
)
from .features.continuous import ContinuousFeatures
from .features.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    SequenceEmbeddingFeatures,
)


def inputs(
    schema: Schema,
    branches: Optional[Dict[str, Block]] = None,
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = None,
    seq: bool = False,
    add_continuous_branch: bool = True,
    continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
    continuous_projection: Optional[Block] = None,
    add_embedding_branch: bool = True,
    embedding_options: EmbeddingOptions = EmbeddingOptions(),
    categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
    **kwargs,
) -> Block:
    branches = branches or {}

    if add_continuous_branch and schema.select_by_tag(continuous_tags).column_schemas:
        branches["continuous"] = ContinuousFeatures.from_schema(
            schema,
            tags=continuous_tags,
        )
    if add_embedding_branch and schema.select_by_tag(categorical_tags).column_schemas:
        emb_cls = SequenceEmbeddingFeatures if seq else EmbeddingFeatures

        branches["categorical"] = emb_cls.from_schema(
            schema, tags=categorical_tags, options=embedding_options
        )

    if continuous_projection:
        return ContinuousEmbedding(
            ParallelBlock(branches),
            continuous_projection,
            aggregation=aggregation,
            post=post,
            name="continuous_projection",
        )

    return ParallelBlock(branches, aggregation=aggregation, post=post, **kwargs)


def prediction_tasks(
    schema: Schema,
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
    task_weight_dict: Optional[Dict[str, float]] = None,
    bias_block: Optional[Layer] = None,
    loss_reduction=tf.reduce_mean,
    **kwargs,
) -> ParallelPredictionBlock:
    return ParallelPredictionBlock.from_schema(
        schema,
        task_blocks=task_blocks,
        task_weight_dict=task_weight_dict,
        bias_block=bias_block,
        loss_reduction=loss_reduction,
        **kwargs,
    )


def sequential(
    *blocks,
    filter: Optional[Union[Schema, Tag, List[str], Filter]] = None,
    block_name: Optional[str] = None,
    copy_layers: bool = False,
    pre_aggregation: Optional[TabularAggregationType] = None,
    **kwargs,
) -> SequentialBlock:
    if pre_aggregation:
        blocks = [TabularBlock(aggregation=pre_aggregation), *blocks]

    return SequentialBlock(
        blocks, filter=filter, block_name=block_name, copy_layers=copy_layers, **kwargs
    )


def merge(
    *branches: Union[Block, Dict[str, Block]],
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = None,
    **kwargs,
) -> ParallelBlock:
    return ParallelBlock(*branches, post=post, aggregation=aggregation, **kwargs)
