from typing import Dict, Optional, Tuple, Type, Union, List

from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType

from ..core import Block, Filter, ParallelBlock, SequentialBlock, TabularBlock
from ..features.continuous import ContinuousFeatures
from ..features.embedding import EmbeddingFeatures


def ContinuousEmbedding(
    inputs: Block,
    embedding_block: Block,
    aggregation=None,
    continuous_aggregation="concat",
    **kwargs
) -> SequentialBlock:
    continuous_embedding = Filter(Tag.CONTINUOUS, aggregation=continuous_aggregation).apply(
        embedding_block
    )

    outputs = inputs.branch(
        continuous_embedding.as_tabular("continuous"),
        add_rest=True,
        aggregation=aggregation,
        **kwargs
    )

    return outputs


def TabularFeatures(
    schema: Schema,
    extra_branches: Optional[Dict[str, Block]] = None,
    continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
    categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
    aggregation: Optional[str] = None,
    continuous_projection: Optional[Block] = None,
    embedding_dim_default: Optional[int] = 64,
    continuous_module_cls: Type[TabularBlock] = ContinuousFeatures,
    embedding_module_cls: Type[TabularBlock] = EmbeddingFeatures,
    add_to_context: List[Union[str, Tag]] = None,
) -> Block:
    branches = extra_branches or {}
    if continuous_tags:
        maybe_continuous_layer = continuous_module_cls.from_schema(
            schema,
            tags=continuous_tags,
        )
        if maybe_continuous_layer:
            branches["continuous"] = maybe_continuous_layer
    if categorical_tags:
        maybe_categorical_layer = embedding_module_cls.from_schema(
            schema, tags=categorical_tags, embedding_dim_default=embedding_dim_default
        )
        if maybe_categorical_layer:
            branches["categorical"] = maybe_categorical_layer
    if add_to_context:
        for item in add_to_context:
            branches[str(item)] = Filter(item, add_to_context=True)

    if continuous_projection:
        inputs = ContinuousEmbedding(ParallelBlock(branches), continuous_projection)
    else:
        inputs = ParallelBlock(branches, aggregation=aggregation)

    return inputs
