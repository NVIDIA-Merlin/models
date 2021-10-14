from typing import Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema

from ..tabular.aggregation import StackFeatures
from ..tabular.base import (
    Block,
    ParallelBlock,
    TabularAggregationType,
    TabularBlock,
    TabularTransformationType,
)


class MultiExpertsBlock(ParallelBlock):
    def __init__(
        self,
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_experts: int,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = StackFeatures(axis=1),
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        experts = {f"expert_{i}": self.create_expert(expert_block, i) for i in range(num_experts)}

        super().__init__(
            experts,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=False,
            **kwargs,
        )

    @staticmethod
    def create_expert(expert_block, i) -> TabularBlock:
        if not isinstance(expert_block, Block):
            expert_block = Block.from_layer(expert_block)

        return expert_block.as_tabular(f"expert_{i}")


class MMOEGate(tf.keras.layers.Layer):
    def __init__(
        self,
        num_experts: int,
        dim=32,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.gate = tf.keras.layers.Dense(dim, name=f"gate_{name}")
        self.softmax = tf.keras.layers.Dense(
            num_experts, use_bias=False, activation="softmax", name=f"gate_distribution_{name}"
        )

    def call(self, body_outputs, expert_outputs, **kwargs):  # type: ignore
        expanded_gate_output = tf.expand_dims(self.softmax(self.gate(body_outputs)), axis=-1)

        out = tf.reduce_sum(expert_outputs * expanded_gate_output, axis=1, keepdims=False)

        return out
