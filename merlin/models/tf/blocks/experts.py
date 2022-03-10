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
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from merlin.models.tf.blocks.core.aggregation import StackFeatures
from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.typing import TabularData
from merlin.schema import Schema


class MMOEGate(Block):
    def __init__(self, num_experts: int, dim=32, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_experts = num_experts

        self.gate = tf.keras.layers.Dense(dim, name=f"gate_{name}")
        self.softmax = tf.keras.layers.Dense(
            num_experts, use_bias=False, activation="softmax", name=f"gate_distribution_{name}"
        )

    def call(self, inputs: TabularData, **kwargs):
        shortcut = inputs.pop("shortcut")
        expert_outputs = list(inputs.values())[0]

        expanded_gate_output = tf.expand_dims(self.softmax(self.gate(shortcut)), axis=-1)
        out = tf.reduce_sum(expert_outputs * expanded_gate_output, axis=1, keepdims=False)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape["shortcut"]

    def get_config(self):
        config = super().get_config()
        config.update(dim=self.dim, num_experts=self.num_experts)

        return config


def MMOEBlock(
    outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock],
    expert_block: Block,
    num_experts: int,
    gate_dim: int = 32,
):
    if isinstance(outputs, ParallelPredictionBlock):
        output_names = outputs.task_names
    elif all(isinstance(x, PredictionTask) for x in outputs):
        output_names = [o.task_name for o in outputs]  # type: ignore
    else:
        output_names = outputs  # type: ignore

    experts = expert_block.repeat_in_parallel(
        num_experts, prefix="expert_", aggregation=StackFeatures(axis=1)
    )
    gates = MMOEGate(num_experts, dim=gate_dim).repeat_in_parallel(names=output_names)
    mmoe = expert_block.connect_with_shortcut(experts, block_outputs_name="experts")
    mmoe = mmoe.connect(gates, block_name="MMOE")

    return mmoe


class CGCGateTransformation(TabularBlock):
    def __init__(
        self,
        task_names: List[str],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        add_shared_gate: bool = True,
        dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        num_total_experts = num_task_experts + num_shared_experts
        self.task_names = [*task_names, "shared"] if add_shared_gate else task_names
        self.stack = StackFeatures(axis=1)
        self.gate_dict: Dict[str, MMOEGate] = {
            name: MMOEGate(num_total_experts, dim=dim) for name in task_names
        }

        if add_shared_gate:
            self.gate_dict["shared"] = MMOEGate(
                len(task_names) * num_task_experts + num_shared_experts, dim=dim
            )

    def call(self, expert_outputs: TabularData, **kwargs) -> TabularData:  # type: ignore
        outputs: TabularData = {}

        shortcut = expert_outputs.pop("shortcut")
        outputs["shortcut"] = shortcut

        for name in self.task_names:
            experts = dict(
                experts=self.stack(self.filter_expert_outputs(expert_outputs, name)),
                shortcut=shortcut,
            )
            outputs[name] = self.gate_dict[name](experts)

        return outputs

    def filter_expert_outputs(self, expert_outputs: TabularData, task_name: str) -> TabularData:
        if task_name == "shared":
            return expert_outputs

        filtered_experts: TabularData = {}
        for name, val in expert_outputs.items():
            if name.startswith((task_name, "shared")):
                filtered_experts[name] = val

        return filtered_experts

    def compute_output_shape(self, input_shape):
        tensor_output_shape = list(input_shape.values())[0]

        return {name: tensor_output_shape for name in self.task_names}


class CGCBlock(ParallelBlock):
    def __init__(
        self,
        outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock],
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        add_shared_gate: bool = True,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(expert_block, Block):
            expert_block = Block.from_layer(expert_block)

        if isinstance(outputs, ParallelPredictionBlock):
            output_names = outputs.task_names
        elif all(isinstance(x, PredictionTask) for x in outputs):
            output_names = [o.task_name for o in outputs]  # type: ignore
        else:
            output_names = outputs  # type: ignore
        task_experts = dict(
            [
                create_expert(expert_block, f"{task}/expert_{i}")
                for task in output_names
                for i in range(num_task_experts)
            ]
        )

        shared_experts = dict(
            [create_expert(expert_block, f"shared/expert_{i}") for i in range(num_shared_experts)]
        )

        post = CGCGateTransformation(
            output_names, num_task_experts, num_shared_experts, add_shared_gate=add_shared_gate
        )
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
            outputs = dict(shortcut=inputs["shortcut"])
            for name, layer in self.parallel_dict.items():
                input_name = "/".join(name.split("/")[:-1])
                outputs.update(layer(inputs[input_name]))

            return outputs
        else:
            outputs = super().call(inputs, **kwargs)
            outputs["shortcut"] = inputs  # type: ignore

        return outputs

    def compute_call_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            output_shapes = {}

            for name, layer in self.parallel_dict.items():
                input_name = "/".join(name.split("/")[:-1])
                output_shapes.update(layer.compute_output_shape(input_shape[input_name]))

            return output_shapes

        return super().compute_call_output_shape(input_shape)


def create_expert(expert_block: Block, name: str) -> Tuple[str, TabularBlock]:
    return name, expert_block.as_tabular(name)
