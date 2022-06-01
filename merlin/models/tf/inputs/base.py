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
from typing import Dict, Optional, Tuple, Type, Union

from merlin.models.tf.blocks.core.aggregation import SequenceAggregation, SequenceAggregator
from merlin.models.tf.blocks.core.base import Block, BlockType
from merlin.models.tf.blocks.core.combinators import ParallelBlock, TabularAggregationType
from merlin.models.tf.blocks.core.masking import MaskingBlock, masking_registry
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    SequenceEmbeddingFeatures,
)
from merlin.schema import Schema, Tags, TagsType

LOG = logging.getLogger("merlin-models")


def InputBlock(
    schema: Schema,
    branches: Optional[Dict[str, Block]] = None,
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = None,
    seq: bool = False,
    max_seq_length: Optional[int] = None,
    add_continuous_branch: bool = True,
    continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
    continuous_projection: Optional[Block] = None,
    add_embedding_branch: bool = True,
    embedding_options: EmbeddingOptions = EmbeddingOptions(),
    categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
    sequential_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.SEQUENCE,),
    split_sparse: bool = False,
    masking: Optional[Union[str, MaskingBlock]] = None,
    seq_aggregator: Block = SequenceAggregator(SequenceAggregation.MEAN),
    **kwargs,
) -> Block:
    """The entry block of the model to process input features from a schema.

    This function creates continuous and embedding layers, and connects them via `ParallelBlock`.
        If aggregation argument is not set, it returns a dictionary of multiple tensors
        each corresponds to an input feature.
        Otherwise, it merges the tensors into one using the aggregation method.

    Example usage::

        mlp = ml.InputBlock(schema).connect(ml.MLPBlock([64, 32]))

    Parameters:
    ----------
    schema: Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    branches: Dict[str, Block], optional
        Dictionary of branches to use inside the InputBlock.
    post: Optional[BlockType]
        Transformations to apply on the inputs after the module is
        called (so **after** `forward`).
        Defaults to None
    aggregation: Optional[TabularAggregationType]
        Aggregation to apply after processing the  `forward`-method to output a single Tensor.
        Defaults to None
    seq: bool
        Whether to process inputs for sequential model (returns 3-D tensor)
        or not (returns 2-D tensor). Use `seq=True` to treat the sparse (list) features
        as sequences (e.g. for sequential recommendation) and `seq=False` to treat sparse
        features as multi-hot categorical representations.
        Defaults to False
    add_continuous_branch: bool
        If set, add the branch to process continuous features
        Defaults to True
    continuous_tags: Optional[Union[TagsType, Tuple[Tags]]]
        Tags to filter the continuous features
        Defaults to  (Tags.CONTINUOUS,)
    continuous_projection: Optional[Block]
        If set, concatenate all numerical features and projet using the
        specified Block.
        Defaults to None
    add_embedding_branch: bool
        If set, add the branch to process categorical features
        Defaults to True
    embedding_options : EmbeddingOptions, optional
        An EmbeddingOptions instance, which allows for a number of
        options for the embedding table, by default EmbeddingOptions()
    categorical_tags: Optional[Union[TagsType, Tuple[Tags]]]
        Tags to filter the continuous features
        Defaults to (Tags.CATEGORICAL,)
    sequential_tags: Optional[Union[TagsType, Tuple[Tags]]]
        Tags to filter the sparse features
        Defaults to (Tags.SEQUENCE,)
    split_sparse: Optional[bool]
        When True, separate the processing of context (2-D) and sparse features (3-D).
        Defaults to False
    masking: Optional[Union[str, MaskSequence]], optional
        If set, Apply masking to the input embeddings and compute masked labels.
        Defaults to None
    seq_aggregator: Block
        If non-sequential model (seq=False):
        aggregate the sparse features tensor along the sequence axis.
        Defaults to SequenceAggregator('mean')
    """
    branches = branches or {}

    if split_sparse:
        sparse_schema = schema.select_by_tag(sequential_tags)
        context_schema = schema.remove_by_tag(sequential_tags)
        if not sparse_schema:
            raise ValueError(
                "Please make sure that schema has features tagged as 'sequence' when"
                "`split_context` is set to True"
            )
        if not aggregation:
            LOG.info(
                "aggregation is not provided, "
                "default `concat` will be used to merge sequential features"
            )
            aggregation = "concat"
        agg = aggregation
        sparse_interactions = InputBlock(
            sparse_schema,
            branches,
            post,
            aggregation=agg,
            seq=True,
            max_seq_length=max_seq_length,
            add_continuous_branch=add_continuous_branch,
            continuous_tags=continuous_tags,
            continuous_projection=continuous_projection,
            add_embedding_branch=add_embedding_branch,
            embedding_options=embedding_options,
            categorical_tags=categorical_tags,
            split_sparse=False,
        )
        if masking:
            if isinstance(masking, str):
                masking = masking_registry.parse(masking)()
            sparse_interactions = sparse_interactions.connect(masking)

        if not seq:
            sparse_interactions = sparse_interactions.connect(seq_aggregator)

        if not context_schema:
            return sparse_interactions

        branches["sparse"] = sparse_interactions
        return InputBlock(
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
            split_sparse=False,
        )

    if (
        add_continuous_branch
        and schema.select_by_tag(continuous_tags).excluding_by_tag(Tags.TARGET).column_schemas
    ):
        pre = None
        if max_seq_length and seq:
            pre = AsDenseFeatures(max_seq_length)
        branches["continuous"] = ContinuousFeatures.from_schema(  # type: ignore
            schema,
            tags=continuous_tags,
            pre=pre,
        )
    if (
        add_embedding_branch
        and schema.select_by_tag(categorical_tags).excluding_by_tag(Tags.TARGET).column_schemas
    ):
        emb_cls: Type[EmbeddingFeatures] = SequenceEmbeddingFeatures if seq else EmbeddingFeatures
        emb_kwargs = {}
        if max_seq_length and seq:
            emb_kwargs["max_seq_length"] = max_seq_length

        branches["categorical"] = emb_cls.from_schema(  # type: ignore
            schema, tags=categorical_tags, embedding_options=embedding_options, **emb_kwargs
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
