import abc
from typing import List, Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag

from ..core import TabularAggregation, TabularBlock, TabularTransformationType
from ..features.embedding import EmbeddingFeatures
from ..typing import TabularData
from .dual import DualEncoderBlock
from .mlp import MLPBlock


class Distance(TabularAggregation, abc.ABC):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        assert len(inputs) == 2

        return self.distance(inputs, **kwargs)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()


class CosineSimilarity(Distance):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        return self.dot(list(inputs.values()))


class Retrieval(DualEncoderBlock):
    def __init__(
        self,
        query: Union[tf.keras.layers.Layer, TabularBlock],
        item: Union[tf.keras.layers.Layer, TabularBlock],
        distance: Distance = CosineSimilarity(),
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        left_name="user",
        right_name="item",
        **kwargs
    ):
        super().__init__(
            query,
            item,
            pre=pre,
            post=post,
            aggregation=distance,
            schema=schema,
            name=name,
            left_name=left_name,
            right_name=right_name,
            **kwargs
        )

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        dims: List[int],
        query_tower_tag=Tag.USER,
        item_tower_tag=Tag.ITEM,
        **kwargs
    ) -> "Retrieval":
        query_tower = MLPBlock.from_schema(schema.select_by_tag(query_tower_tag), dims)
        item_tower = MLPBlock.from_schema(schema.select_by_tag(item_tower_tag), dims)

        return cls(query_tower, item_tower, **kwargs)


class MatrixFactorization(DualEncoderBlock):
    def __init__(
        self,
        query_embedding: EmbeddingFeatures,
        item_embedding: EmbeddingFeatures,
        distance: Distance = CosineSimilarity(),
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        left_name="user",
        right_name="item",
        **kwargs
    ):
        super().__init__(
            query_embedding,
            item_embedding,
            pre=pre,
            post=post,
            aggregation=distance,
            schema=schema,
            name=name,
            left_name=left_name,
            right_name=right_name,
            **kwargs
        )

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        query_id_tag=Tag.USER_ID,
        item_id_tag=Tag.ITEM_ID,
        distance: Distance = CosineSimilarity(),
        **kwargs
    ) -> "MatrixFactorization":
        query = EmbeddingFeatures.from_schema(schema.select_by_tag(query_id_tag))
        item = EmbeddingFeatures.from_schema(schema.select_by_tag(item_id_tag))

        return cls(query, item, distance=distance, **kwargs)
