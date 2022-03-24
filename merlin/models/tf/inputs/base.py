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
from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union, Any

from merlin.models.utils import schema_utils

from merlin.models.tf.blocks.core.aggregation import SequenceAggregation, SequenceAggregator
from merlin.models.tf.blocks.core.base import Block, BlockType
from merlin.models.tf.blocks.core.combinators import ParallelBlock, TabularAggregationType
from merlin.models.tf.blocks.core.masking import MaskingBlock, masking_registry
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures, CategoricalOneHot
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import ContinuousEmbedding, EmbeddingFeatures, EmbeddingOptions
from merlin.schema import Schema, Tags, TagsType

LOG = logging.getLogger("merlin-models")


def InputBlock(
        schema: Schema,
        branches: Optional[Dict[str, Block]] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        seq: bool = False,
        max_seq_length: Optional[int] = None,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        add_continuous_branch: bool = True,
        continuous_projection: Optional[Block] = None,
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        add_embedding_branch: bool = True,
        embedding_options: Optional[EmbeddingOptions] = None,
        sequence_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.SEQUENCE,),
        add_sequence_branch: bool = False,
        seq_aggregator: Optional[Union[str, SequenceAggregation, Block]] = None,
        masking: Optional[Union[str, MaskingBlock]] = None,
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
    max_seq_length: Optional[int]
        Maximum sequence length to use for sequence features.
        This will be inferred from the schema if not set and `seq=True`.
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
        If true, add an embedding-table for each categorical features.
        If false, One-hot encoding is used to represent each categorical feature instead.
        Defaults to True
    categorical_tags: Optional[Union[TagsType, Tuple[Tags]]]
        Tags to filter the continuous features
        Defaults to (Tags.CATEGORICAL,)
    add_sequence_branch: bool
        When True, separate the processing of context (2-D) and sparse features (3-D).
        Defaults to False. The tags to use to split the sequence are specified in `sequential_tags`.
    sequence_tags: Optional[Union[TagsType, Tuple[Tags]]]
        Tags to use when splitting the sequence features.
        Defaults to (Tags.SEQUENCE,)
    masking: Optional[Union[str, MaskSequence]], optional
        If set, Apply masking to the input embeddings and compute masked labels.
        Defaults to None
    seq_aggregator: Block
        If non-sequential model (seq=False):
        aggregate the sparse features' tensor along the sequence axis.
        Defaults to SequenceAggregator("mean")
    """
    _branches: Dict[str, Block] = branches or {}
    if embedding_options:
        _embedding_options = embedding_options
    else:
        _embedding_options = EmbeddingOptions(schema)

    if add_sequence_branch and schema.select_by_tag(sequence_tags).column_schemas:
        _params_dict: Dict[str, Any] = locals()  # This contains all input parameters
        _params_dict["branches"] = _branches
        _params_dict["embedding_options"] = _embedding_options
        for to_del in ["add_sequence_branch", "kwargs"]:
            del _params_dict[to_del]

        return SequentialInputBlockWithContext(**_params_dict, **kwargs)

    if add_continuous_branch and schema.select_by_tag(continuous_tags).column_schemas:
        _branches[InputBranches.CONTINUOUS.value] = ContinuousFeatures(
            schema.select_by_tag(continuous_tags),
            max_seq_length=max_seq_length,
        )
    if schema.select_by_tag(categorical_tags).column_schemas:
        if add_embedding_branch:
            _branches[InputBranches.CATEGORICAL.value] = EmbeddingFeatures(embedding_options)
        else:
            _branches[InputBranches.CATEGORICAL.value] = CategoricalOneHot(schema.select_by_tag(categorical_tags))

    out_kwargs = dict(post=post, aggregation=aggregation)
    if continuous_projection:
        return ContinuousEmbedding(_branches, continuous_projection, **out_kwargs)

    return ParallelBlock(_branches, is_input=True, **out_kwargs, **kwargs)


class InputBranches(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    SEQUENCE = "sequence"


def SequentialInputBlock(
        schema: Schema,
        max_seq_length: Optional[int] = None,
        seq_aggregator: Optional[Union[str, SequenceAggregation, Block]] = None,
        branches: Optional[Dict[str, Block]] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        add_continuous_branch: bool = True,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        continuous_projection: Optional[Block] = None,
        add_embedding_branch: bool = True,
        embedding_options: Optional[EmbeddingOptions] = None,
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        masking: Optional[Union[str, MaskingBlock]] = None,
) -> InputBlock:
    if not max_seq_length:
        max_seq_length = schema_utils.max_value_count(schema)

        if not max_seq_length:
            raise ValueError("max_seq_length couldn't be inferred, please provide it explicitly.")

    if embedding_options is None:
        embedding_options = EmbeddingOptions(schema)
    else:
        embedding_options = embedding_options.select_by_schema(schema)
    embedding_options.max_seq_length = max_seq_length
    _params_dict = locals()
    _masking = _params_dict.pop("masking", None)

    block = InputBlock(seq=True, split_sequence=False, **_params_dict)
    if _masking:
        if isinstance(_masking, str):
            _masking = masking_registry.parse(_masking)()
        block = block.connect(_masking)

    if not seq_aggregator:
        block = block.connect(SequenceAggregation.parse(seq_aggregator))

    return block


def SequentialInputBlockWithContext(
        schema: Schema,
        sequence_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.SEQUENCE,),
        max_seq_length: Optional[int] = None,
        seq_aggregator: Optional[Union[str, SequenceAggregation, Block]] = None,
        branches: Optional[Dict[str, Block]] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        add_continuous_branch: bool = True,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        continuous_projection: Optional[Block] = None,
        add_embedding_branch: bool = True,
        embedding_options: Optional[EmbeddingOptions] = None,
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        masking: Optional[Union[str, MaskingBlock]] = None,
) -> InputBlock:
    sequence_schema = schema.select_by_tag(sequence_tags)
    context_schema = schema.remove_by_tag(sequence_tags)

    if not sequence_schema:
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

    input_kwargs = dict(
        branches=branches,
        post=post,
        aggregation=aggregation,
        add_continuous_branch=add_continuous_branch,
        continuous_tags=continuous_tags,
        continuous_projection=continuous_projection,
        add_embedding_branch=add_embedding_branch,
        categorical_tags=categorical_tags,
    )

    sparse_branch = SequentialInputBlock(
        sequence_schema,
        max_seq_length,
        embedding_options=embedding_options,
        masking=masking,
        seq_aggregator=seq_aggregator,
        **input_kwargs
    )

    if not context_schema:
        return sparse_branch

    branches[InputBranches.SEQUENCE.value] = sparse_branch

    return InputBlock(
        context_schema,
        branches=branches,
        embedding_options=embedding_options,
        seq=False,
        split_sparse=False,
        **input_kwargs
    )
