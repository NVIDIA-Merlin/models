import abc
from typing import Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema

from merlin_models.tf import MergeTabular, SequentialBlock, TabularBlock
from merlin_models.tf.tabular.base import AsTabular, TabularAggregation, TabularTransformationType
from merlin_models.tf.typing import TabularData


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


class Retrieval(MergeTabular):
    def __init__(
        self,
        left: Union[tf.keras.layers.Layer, TabularBlock],
        right: Union[tf.keras.layers.Layer, TabularBlock],
        distance: Distance = CosineSimilarity(),
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        left_name="user",
        right_name="item",
        **kwargs
    ):
        if not isinstance(left, TabularBlock):
            left = SequentialBlock([left, AsTabular(left_name)])
        if not isinstance(right, TabularBlock):
            right = SequentialBlock([right, AsTabular(right_name)])

        super().__init__(
            left,
            right,
            pre=pre,
            post=post,
            aggregation=distance,
            schema=schema,
            name=name,
            **kwargs
        )
