import abc
from typing import Optional, List, Text, Tuple, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag

from .inputs import TabularFeatures
from ..core import (
    Block,
    ParallelBlock,
    TabularAggregation,
    inputs,
    merge,
    tabular_aggregation_registry,
    TabularTransformationsType,
    PredictionTask,
    MetricOrMetricClass, Sampler, SequentialBlock,
)
from ..features.embedding import EmbeddingFeatures
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
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        out = self.dot(list(inputs.values()))

        return out


def TwoTowerBlock(
        schema,
        query_tower: Block,
        item_tower: Optional[Block] = None,
        query_tower_tag=Tag.USER,
        item_tower_tag=Tag.ITEM,
        add_to_query_context: List[Union[str, Tag]] = None,
        add_to_item_context: List[Union[str, Tag]] = None,
        embedding_dim_default: Optional[int] = 64,
        post: Optional[TabularTransformationsType] = None,
        # negative_memory_bank=None,
        **kwargs
) -> ParallelBlock:
    _item_tower: Block = item_tower or query_tower.copy()
    if not isinstance(query_tower, SequentialBlock) and not query_tower.inputs:
        query_tower = TabularFeatures(
            schema.select_by_tag(query_tower_tag),
            embedding_dim_default=embedding_dim_default,
            add_to_context=add_to_query_context
        ).apply(query_tower)
    if not isinstance(_item_tower, SequentialBlock) and not _item_tower.inputs:
        _item_tower = TabularFeatures(
            schema.select_by_tag(item_tower_tag),
            embedding_dim_default=embedding_dim_default,
            add_to_context=add_to_item_context
        ).apply(item_tower)

    two_tower = ParallelBlock(
        {str(query_tower_tag): query_tower, str(item_tower_tag): _item_tower},
        post=post,
        **kwargs
    )

    return two_tower


def MatrixFactorizationBlock(
        schema: Schema,
        dim: int,
        query_id_tag=Tag.USER_ID,
        item_id_tag=Tag.ITEM_ID,
        distance="cosine",
        **kwargs
):
    query_id, item_id = schema.select_by_tag(query_id_tag), schema.select_by_tag(item_id_tag)
    matrix_factorization = merge(
        {
            str(query_id_tag): EmbeddingFeatures.from_schema(query_id, embedding_dim_default=dim),
            str(item_id_tag): EmbeddingFeatures.from_schema(item_id, embedding_dim_default=dim),
        },
        aggregation=distance,
        **kwargs
    )

    return matrix_factorization


# class TwoTower(DualEncoderBlock):
#     def __init__(
#             self,
#             query: Union[tf.keras.layers.Layer, Block],
#             item: Union[tf.keras.layers.Layer, Block],
#             distance: Distance = CosineSimilarity(),
#             pre: Optional[TabularTransformationType] = None,
#             post: Optional[TabularTransformationType] = None,
#             schema: Optional[Schema] = None,
#             name: Optional[str] = None,
#             left_name="query",
#             right_name="item",
#             **kwargs
#     ):
#         super().__init__(
#             query,
#             item,
#             pre=pre,
#             post=post,
#             aggregation=distance,
#             schema=schema,
#             name=name,
#             left_name=left_name,
#             right_name=right_name,
#             **kwargs
#         )
#         self.left_name = left_name
#         self.right_name = right_name
#
#     @classmethod
#     def from_schema(  # type: ignore
#             cls,
#             schema: Schema,
#             dims: List[int],
#             query_tower_tag=Tag.USER,
#             item_tower_tag=Tag.ITEM,
#             **kwargs
#     ) -> "TwoTower":
#         query_tower = inputs(schema.select_by_tag(query_tower_tag), MLPBlock(dims))
#         item_tower = inputs(schema.select_by_tag(item_tower_tag), MLPBlock(dims))
#
#         return cls(query_tower, item_tower, **kwargs)
#
#     @property
#     def query_tower(self) -> Optional[tf.keras.layers.Layer]:
#         if self.left_name in self.parallel_dict:
#             return self.parallel_dict[self.left_name]
#
#         return None
#
#     @property
#     def item_tower(self) -> Optional[tf.keras.layers.Layer]:
#         if self.right_name in self.parallel_dict:
#             return self.parallel_dict[self.right_name]
#
#         return None
#
#
# class MatrixFactorization(DualEncoderBlock):
#     def __init__(
#             self,
#             query_embedding: EmbeddingFeatures,
#             item_embedding: EmbeddingFeatures,
#             distance: Distance = CosineSimilarity(),
#             pre: Optional[TabularTransformationType] = None,
#             post: Optional[TabularTransformationType] = None,
#             schema: Optional[Schema] = None,
#             name: Optional[str] = None,
#             left_name="user",
#             right_name="item",
#             **kwargs
#     ):
#         super().__init__(
#             query_embedding,
#             item_embedding,
#             pre=pre,
#             post=post,
#             aggregation=distance,
#             schema=schema,
#             name=name,
#             left_name=left_name,
#             right_name=right_name,
#             **kwargs
#         )
#
#     @classmethod
#     def from_schema(
#             cls,
#             schema: Schema,
#             query_id_tag=Tag.USER_ID,
#             item_id_tag=Tag.ITEM_ID,
#             distance: Distance = CosineSimilarity(),
#             **kwargs
#     ) -> "MatrixFactorization":
#         query = EmbeddingFeatures.from_schema(schema.select_by_tag(query_id_tag))
#         item = EmbeddingFeatures.from_schema(schema.select_by_tag(item_id_tag))
#
#         return cls(query, item, distance=distance, **kwargs)
