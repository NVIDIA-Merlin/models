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


from typing import Dict, Optional, Tuple, Union

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType

from ..core import Block, BlockType, ParallelBlock, TabularAggregationType
from ..features.continuous import ContinuousFeatures
from ..features.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    SequenceEmbeddingFeatures,
)


def InputBlock(
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
