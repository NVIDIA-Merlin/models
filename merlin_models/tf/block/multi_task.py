from typing import Optional, Union, List, Dict

import tensorflow as tf
from merlin_standard_lib import Schema

from ..model.base import PredictionTask
from ..tabular.aggregation import StackFeatures
from ..tabular.base import (
    Block,
    ParallelBlock,
    TabularAggregationType,
    TabularBlock,
    TabularTransformationType, TabularTransformation,
)
from ..typing import TabularData


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
        experts = {f"expert_{i}": create_expert(expert_block, i) for i in range(num_experts)}

        super().__init__(
            experts,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=False,
            **kwargs,
        )


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


class CGCGateTransformation(TabularTransformation):
    def __init__(
            self,
            task_names: List[str],
            num_task_experts: int = 1,
            num_shared_experts: int = 1,
            is_last: bool = False,
            dim: int = 32,
            **kwargs):
        super().__init__(**kwargs)
        num_total_experts = num_task_experts + num_shared_experts
        self.task_names = task_names if is_last else [*task_names, "shared"]
        self.stack = StackFeatures(axis=1)
        self.gate_dict: Dict[str, MMOEGate] = {name: MMOEGate(num_total_experts, dim=dim)
                                               for name in task_names}

    def call(
            self,
            expert_outputs: TabularData,
            body_outputs: tf.Tensor,
            **kwargs
    ) -> TabularData:  # type: ignore
        outputs: TabularData = {}

        for name in self.task_names:
            experts = self.stack(self.filtered_experts(expert_outputs, name))
            outputs[name] = self.gate_dict[name](body_outputs, experts)

        return outputs

    def filter_expert_outputs(self, expert_outputs: TabularData, task_name: str) -> TabularData:
        filtered_experts: TabularData = {}
        for name, val in expert_outputs.items():
            if name.startswith((task_name, "shared")):
                filtered_experts[name] = val

        return filtered_experts


class CGCBlock(ParallelBlock):
    def __init__(
            self,
            expert_block: Union[Block, tf.keras.layers.Layer],
            prediction_tasks: List[PredictionTask],
            num_task_experts: int = 1,
            num_shared_experts: int = 1,
            is_last: bool = False,
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            **kwargs,
    ):
        self.is_last = is_last
        task_names: List[str] = [task.task_name for task in prediction_tasks]
        task_experts = {f"{task}/expert_{i}": create_expert(expert_block, i)
                        for task in task_names for i in range(num_task_experts)}
        shared_experts = {f"shared/expert_{i}": create_expert(expert_block, i)
                          for i in range(num_shared_experts)}

        post = CGCGateTransformation(task_names, num_task_experts, num_shared_experts,
                                     is_last=is_last)
        super().__init__(
            task_experts,
            shared_experts,
            post=post,
            aggregation=None,
            schema=schema,
            name=name,
            strict=False,
            **kwargs,
        )

    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            outputs = {}
            for name, layer in self.to_merge_dict:
                input_name = name.split("/")[0]
                outputs.update(layer(inputs[input_name]))

            return outputs

        return super().call(inputs, **kwargs)


def create_expert(expert_block, i) -> TabularBlock:
    if not isinstance(expert_block, Block):
        expert_block = Block.from_layer(expert_block)

    return expert_block.as_tabular(f"expert_{i}")
