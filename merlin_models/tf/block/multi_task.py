from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from merlin_standard_lib import Schema

from ..core import (
    Block, ParallelBlock, PredictionTask, TabularBlock, TabularTransformation, TabularAggregation
)
from ..tabular.aggregation import StackFeatures
from ..typing import TabularData


class MMOEGate(Block):
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
        self.dim = dim
        self.num_experts = num_experts

        self.gate = tf.keras.layers.Dense(dim, name=f"gate_{name}")
        self.softmax = tf.keras.layers.Dense(
            num_experts, use_bias=False, activation="softmax", name=f"gate_distribution_{name}"
        )

    def call(self, inputs: TabularData, **kwargs):  # type: ignore
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


class MMOEGateAggregation(TabularAggregation):
    def __init__(
        self,
        num_experts: int,
        dim=32,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gate = tf.keras.layers.Dense(dim, name=f"gate_{name}")
        self.stack = StackFeatures(axis=1)
        self.softmax = tf.keras.layers.Dense(
            num_experts, use_bias=False, activation="softmax", name=f"gate_distribution_{name}"
        )

    def call(self, inputs, **kwargs):  # type: ignore
        shortcut = inputs.pop("shortcut")
        expanded_gate_output = tf.expand_dims(self.softmax(self.gate(shortcut)), axis=-1)

        expert_outputs = self.stack(inputs)
        out = tf.reduce_sum(expert_outputs * expanded_gate_output, axis=1, keepdims=False)

        return out

    def compute_output_shape(self, input_shape):
        tensor_output_shape = input_shape
        if isinstance(input_shape, dict):
            tensor_output_shape = list(input_shape.values())[0]

        return {name: tensor_output_shape for name in self.gate_dict}


class MultiGateMixtureOfExperts(TabularTransformation):
    def __init__(
        self,
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_experts: int,
        output_names: List[str],
        gate_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(expert_block, Block):
            expert_block = Block.from_layer(expert_block)

        self.output_names = output_names

        agg = StackFeatures(axis=1)
        experts = expert_block.repeat_in_parallel(num_experts, prefix="expert_", aggregation=agg)
        gates = MMOEGate(num_experts, dim=gate_dim).repeat_in_parallel(names=output_names)
        self.mmoe = expert_block.add_with_shortcut(experts).add(gates, block_name="MMOE")

        self.experts = experts
        self.gates = gates

        str(self.mmoe)

        a = 5

        # self.experts = expert_block.repeat_in_parallel(num_experts, prefix="expert_", residual=True)
        # gates = ParallelBlock({task_name: MMOEGate(num_experts, dim=gate_dim)
        #                        for task_name in output_names})
        # self.experts.add(gates)
        # self.gate_dict: Dict[str, MMOEGate] = {}
        # for task_name in output_names:
        #     self.gate_dict[task_name] = MMOEGate(num_experts, dim=gate_dim)

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        experts = self.experts(inputs)
        #
        # gates_inputs = dict(experts=experts, shortcut=inputs)
        # out = self.gates(inputs)
        #
        # return out


        return self.mmoe(inputs)

        # outputs = {}
        #
        # expert_outputs = self.experts(inputs)
        #
        # for name, gate in self.gate_dict.items():
        #     outputs[name] = gate(inputs, expert_outputs)
        #
        # return outputs

    def compute_output_shape(self, input_shape):
        # return self.mmoe.compute_output_shape(input_shape)
        tensor_output_shape = input_shape
        if isinstance(input_shape, dict):
            tensor_output_shape = list(input_shape.values())[0]

        return {name: tensor_output_shape for name in self.output_names}


class CGCGateTransformation(TabularTransformation):
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

    def call(self, expert_outputs: TabularData, **kwargs) -> TabularData:
        outputs: TabularData = {}

        body_outputs = expert_outputs.pop("body_outputs")
        outputs["body_outputs"] = body_outputs

        for name in self.task_names:
            experts = dict(
                experts=self.stack(self.filter_expert_outputs(expert_outputs, name)),
                shortcut=body_outputs
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
        expert_block: Union[Block, tf.keras.layers.Layer],
        prediction_tasks: List[PredictionTask],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        add_shared_gate: bool = True,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(expert_block, Block):
            expert_block = Block.from_layer(expert_block)

        task_names: List[str] = [task.task_name for task in prediction_tasks]
        task_experts = dict(
            [
                create_expert(expert_block, f"{task}/expert_{i}")
                for task in task_names
                for i in range(num_task_experts)
            ]
        )

        shared_experts = dict(
            [create_expert(expert_block, f"shared/expert_{i}") for i in range(num_shared_experts)]
        )

        post = CGCGateTransformation(
            task_names, num_task_experts, num_shared_experts, add_shared_gate=add_shared_gate
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
            outputs = dict(body_outputs=inputs["body_outputs"])
            for name, layer in self.parallel_dict.items():
                input_name = "/".join(name.split("/")[:-1])
                outputs.update(layer(inputs[input_name]))

            return outputs
        else:
            outputs = super().call(inputs, **kwargs)
            outputs["body_outputs"] = inputs  # type: ignore

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
