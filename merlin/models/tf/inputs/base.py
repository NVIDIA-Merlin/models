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
import warnings
from typing import Callable, Dict, Optional, Tuple, Type, Union

from tensorflow.keras.layers import Layer

from merlin.models.tf.core.aggregation import SequenceAggregator
from merlin.models.tf.core.base import Block, BlockType
from merlin.models.tf.core.combinators import ParallelBlock, TabularAggregationType
from merlin.models.tf.inputs.continuous import Continuous, ContinuousFeatures
from merlin.models.tf.inputs.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    Embeddings,
    SequenceEmbeddingFeatures,
)
from merlin.schema import Schema, Tags, TagsType

LOG = logging.getLogger("merlin-models")


def InputBlock(
    schema: Schema,
    branches: Optional[Dict[str, Block]] = None,
    pre: Optional[BlockType] = None,
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
    seq_aggregator: Block = SequenceAggregator("mean"),
    **kwargs,
) -> Block:
    """The entry block of the model to process input features from a schema.

    This function creates continuous and embedding layers, and connects them via `ParallelBlock`.
        If aggregation argument is not set, it returns a dictionary of multiple tensors
        each corresponds to an input feature. Otherwise, it merges the tensors
        into one using the aggregation method.

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
        If set, concatenate all numerical features and project using the
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
    seq_aggregator: Block
        If non-sequential model (seq=False):
        aggregate the sparse features tensor along the sequence axis.
        Defaults to SequenceAggregator('mean')
    """
    # If targets are passed, exclude these from the input block schema
    schema = schema.excluding_by_tag([Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION])

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
            add_continuous_branch=add_continuous_branch,
            continuous_tags=continuous_tags,
            continuous_projection=continuous_projection,
            add_embedding_branch=add_embedding_branch,
            embedding_options=embedding_options,
            categorical_tags=categorical_tags,
            split_sparse=False,
        )

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
        branches["continuous"] = ContinuousFeatures.from_schema(  # type: ignore
            schema, tags=continuous_tags
        )
    if (
        add_embedding_branch
        and schema.select_by_tag(categorical_tags).excluding_by_tag(Tags.TARGET).column_schemas
    ):
        emb_cls: Type[EmbeddingFeatures] = SequenceEmbeddingFeatures if seq else EmbeddingFeatures

        branches["categorical"] = emb_cls.from_schema(  # type: ignore
            schema, tags=categorical_tags, embedding_options=embedding_options
        )
    if continuous_projection:
        return ContinuousEmbedding(
            ParallelBlock(branches),
            continuous_projection,
            aggregation=aggregation,
            post=post,
            name="continuous_projection",
        )

    kwargs["is_input"] = kwargs.get("is_input", True)
    return ParallelBlock(branches, pre=pre, aggregation=aggregation, post=post, **kwargs)


INPUT_TAG_TO_BLOCK: Dict[Tags, Callable[[Schema], Layer]] = {
    Tags.CONTINUOUS: Continuous,
    Tags.CATEGORICAL: Embeddings,
}


def InputBlockV2(
    schema: Optional[Schema] = None,
    categorical: Union[Tags, Layer] = Tags.CATEGORICAL,
    continuous: Union[Tags, Layer] = Tags.CONTINUOUS,
    pre: Optional[BlockType] = None,
    post: Optional[BlockType] = None,
    aggregation: Optional[TabularAggregationType] = "concat",
    tag_to_block=INPUT_TAG_TO_BLOCK,
    **branches,
) -> ParallelBlock:
    """The entry block of the model to process input features from a schema.

    This is a new version of InputBlock, which is more flexible for accepting
    the external definition of `embeddings` block. After `22.10` this will become the default.

    Simple Usage::
        inputs = InputBlockV2(schema)

    Custom Embeddings::
        inputs = InputBlockV2(
            schema,
            categorical=Embeddings(schema, dim=32)
        )

    Sparse outputs for one-hot::
        inputs = InputBlockV2(
            schema,
            categorical=CategoryEncoding(schema, sparse=True),
            post=ToSparse()
        )

    Add continuous projection::
        inputs = InputBlockV2(
            schema,
            continuous=ContinuousProjection(continuous_schema, MLPBlock([32])),
        )

    Merge 2D and 3D (for session-based)::
        inputs = InputBlockV2(
            schema,
            post=BroadcastToSequence(context_schema, sequence_schema)
        )


    Parameters
    ----------
    schema : Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    categorical : Union[Tags, Layer], defaults to `Tags.CATEGORICAL`
        A block or column-selector to use for categorical-features.
        If a column-selector is provided (either a schema or tags), the selector
        will be passed to `Embeddings` to infer the embedding tables from the column-selector.
    continuous : Union[Tags, Layer], defaults to `Tags.CONTINUOUS`
        A block to use for continuous-features.
        If a column-selector is provided (either a schema or tags), the selector
        will be passed to `Continuous` to infer the features from the column-selector.
    pre : Optional[BlockType], optional
        Transformation block to apply before the embeddings lookup, by default None
    post : Optional[BlockType], optional
        Transformation block to apply after the embeddings lookup, by default None
    aggregation : Optional[TabularAggregationType], optional
        Transformation block to apply for aggregating the inputs, by default "concat"
    tag_to_block : Dict[str, Callable[[Schema], Layer]], optional
        Mapping from tag to block-type, by default:
            Tags.CONTINUOUS -> Continuous
            Tags.CATEGORICAL -> Embeddings
    **branches : dict
        Extra branches to add to the input block.

    Returns
    -------
    ParallelBlock
        Returns a ParallelBlock with a Dict with two branches:
        continuous and embeddings
    """
    # If targets are passed, exclude these from the input block schema
    schema = schema.excluding_by_tag([Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION])

    if "embeddings" in branches:
        warnings.warn(
            "The `embeddings` argument is deprecated and should be replaced "
            "by `categorical` argument",
            DeprecationWarning,
        )
        categorical = branches["embeddings"]

    unparsed = {"categorical": categorical, "continuous": continuous, **branches}
    parsed = {}
    for name, branch in unparsed.items():
        if isinstance(branch, Layer):
            parsed[name] = branch
        else:
            if not isinstance(schema, Schema):
                raise ValueError(
                    "If you pass a column-selector as a branch, "
                    "you must also pass a `schema` argument."
                )
            if branch not in tag_to_block:
                raise ValueError(f"No default-block provided for {branch}")
            branch_schema: Schema = schema.select_by_tag(branch)
            if branch_schema:
                parsed[name] = tag_to_block[branch](branch_schema)

    if not parsed:
        raise ValueError("No columns selected for the input block")

    return ParallelBlock(
        parsed,
        pre=pre,
        post=post,
        aggregation=aggregation,
        is_input=True,
        schema=schema,
    )
