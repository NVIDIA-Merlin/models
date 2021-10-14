from typing import Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema

from merlin_models.tf.block.base import SequentialBlock
from merlin_models.tf.tabular.base import (
    AsTabular,
    TabularAggregationType,
    TabularBlock,
    TabularTransformationType,
)
from merlin_models.tf.utils.tf_utils import maybe_serialize_keras_objects


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class DualEncoderBlock(TabularBlock):
    def __init__(
        self,
        left: Union[TabularBlock, tf.keras.layers.Layer],
        right: Union[TabularBlock, tf.keras.layers.Layer],
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        left_name: str = "left",
        right_name: str = "right",
        **kwargs
    ):
        if not getattr(left, "is_tabular", False):
            left = SequentialBlock([left, AsTabular(left_name)])
        if not getattr(right, "is_tabular", False):
            right = SequentialBlock([right, AsTabular(right_name)])

        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )

        self.left = left
        self.right = right

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {}
        for layer in [self.left, self.right]:
            outputs.update(layer(inputs))

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for layer in [self.left, self.right]:
            output_shapes.update(layer.compute_output_shape(input_shape))

        return output_shapes

    def get_config(self):
        return maybe_serialize_keras_objects(
            self, super(DualEncoderBlock, self).get_config(), ["left", "right"]
        )
