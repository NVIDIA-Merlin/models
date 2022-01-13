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
from merlin_standard_lib.utils.proto_utils import has_field

from ..core import Block, BlockType, ParallelBlock, TabularAggregationType
from ..features.continuous import ContinuousFeatures
from ..features.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    SequenceEmbeddingFeatures,
)
from .aggregation import SequenceAggregator
from .masking import MaskingBlock, masking_registry

INPUT_PARAMETERS_DOCSTRING = """
    post: Optional[BlockType]
        Transformations to apply on the inputs after the module is called (so **after** `forward`).
        Defaults to None
    aggregation: Optional[TabularAggregationType]
        Aggregation to apply after processing the  `forward`-method to output a single Tensor.
        Defaults to None
    seq: bool
        Whether to process inputs for sequential model (returns 3-D tensor)
        or not (returns 2-D tensor).
        Defaults to False
    add_continuous_branch: bool
        If set, add the branch to process continuous features
        Defaults to True
    continuous_tags: Optional[Union[TagsType, Tuple[Tag]]]
        Tags to filter the continuous features
        Defaults to  (Tag.CONTINUOUS,)
    continuous_projection: Optional[Block]
        If set, concatenate all numerical features and projet using the
        specified Block.
        Defaults to None
    add_embedding_branch: bool
        If set, add the branch to process categorical features
        Defaults to True
    categorical_tags: Optional[Union[TagsType, Tuple[Tag]]]
        Tags to filter the continuous features
        Defaults to (Tag.CATEGORICAL,)
"""


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
    """Input Block to process categorical and continuous features.

    Parameters:
    ----------
        {INPUT_PARAMETERS_DOCSTRING}
    """

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

    return ParallelBlock(branches, aggregation=aggregation, post=post, is_input=True, **kwargs)


def MixedInputBlock(
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
    masking: Optional[Union[str, MaskingBlock]] = None,
    seq_aggregator: Block = SequenceAggregator("mean"),
    **kwargs,
) -> Block:
    """Input Block to separate the processing of context and sparse features.

    Parameters:
    -----------
        {INPUT_PARAMETERS_DOCSTRING}
        masking: Optional[Union[str, MaskSequence]], optional
            If set, Apply masking to the input embeddings and compute masked labels.
            Defaults to None
        seq_aggregator: Block
            If non-sequential model (seq=False):
            aggregate the sparse features tensor along the sequence axis.
            Defaults to SequenceAggregator('mean')
    """

    sparse_features = [feat.name for feat in schema.feature if has_field(feat, "value_count")]
    sparse_schema = schema.select_by_name(sparse_features)
    context_schema = schema.remove_by_name(sparse_features)

    input_layers = {}

    agg = aggregation if aggregation else "concat"
    if sparse_schema:
        sparse_interactions = InputBlock(
            sparse_schema,
            branches,
            post,
            aggregation=agg,
            seq=True,
            add_continuous_branch=add_continuous_branch,
            continuous_tags=continuous_tags,
            continuous_projection=continuous_projection,
            add_embedding_branch=add_embedding_branch,
            embedding_options=embedding_options,
            categorical_tags=categorical_tags,
        )

        if masking:
            if isinstance(masking, str):
                masking = masking_registry.parse(masking)()
            sparse_interactions = sparse_interactions.connect(masking)

        if not seq:
            sparse_interactions = sparse_interactions.connect(seq_aggregator)

        input_layers["sparse_layer"] = sparse_interactions

    if context_schema:
        context_interactions = InputBlock(
            context_schema,
            branches,
            post,
            aggregation=agg,
            seq=False,
            add_continuous_branch=add_continuous_branch,
            continuous_tags=continuous_tags,
            continuous_projection=continuous_projection,
            add_embedding_branch=add_embedding_branch,
            embedding_options=embedding_options,
            categorical_tags=categorical_tags,
        )
        input_layers["context_layer"] = context_interactions

    if len(input_layers) == 1:
        return input_layers
    else:
        return ParallelBlock(
            input_layers, aggregation=aggregation, post=post, is_input=True, **kwargs
        )
