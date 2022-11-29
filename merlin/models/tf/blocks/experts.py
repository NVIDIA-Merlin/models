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

from merlin.models.tf.core.aggregation import StackFeatures
from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock, TabularBlock
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.typing import TabularData
from merlin.schema import Schema


class MMOEGate(Block):
    """MMoE Gate, which uses input features to generate softmax weights
    in a weighted sum of expert outputs.

    Parameters
    ----------
    num_experts : int
        Number of experts, so that there is a weight for each expert
    gate_block : Block, optional
        Allows for having a Block (e.g. MLPBlock([32])) to combine the inputs
        before the final projection layer (created automatically)
        that outputs a softmax distribution over the number of experts.
        This might give more capacity to the gates to decide from the inputs
        how to better combine the experts.
    softmax_temperature : float, optional
        The temperature of the softmax that is used for weighting the experts outputs,
        by default 1.0
    enable_gate_weights_metric : bool, optional
        Enables logging the average gate weights on experts
    name : str, optional
        Name of the block, by default None
    """

    def __init__(
        self,
        num_experts: int,
        gate_block: Optional[Block] = None,
        softmax_temperature: float = 1.0,
        enable_gate_weights_metrics: bool = False,
        name: str = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.softmax_temperature = softmax_temperature
        self.enable_gate_weights_metrics = enable_gate_weights_metrics
        self.gate_name = name

        self.gate_block = gate_block
        self.gate_final = tf.keras.layers.Dense(
            num_experts, use_bias=False, name=f"gate_final_{name}"
        )

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        """MMOE call

        Parameters
        ----------
        inputs : TabularData
            Expects a dict/TabularData with "shortcut" (inputs) and "experts" blocks

        Returns
        -------
        tf.Tensor
            The experts outputs weighted by the gate
        """
        inputs = dict(inputs)  # Creates a copy of the dict
        if set(inputs.keys()) != set(["shortcut", "experts"]):
            raise ValueError("MMoE gate expects a dict with 'shortcut' and 'experts' keys.")

        inputs_shortcut, expert_outputs = inputs["shortcut"], inputs["experts"]

        inputs_gate = inputs_shortcut
        if self.gate_block is not None:
            inputs_gate = self.gate_block(inputs_gate)
        gate_weights = tf.expand_dims(
            tf.nn.softmax(self.gate_final(inputs_gate) / self.softmax_temperature),
            axis=-1,
            name="gate_softmax",
        )
        out = tf.reduce_sum(expert_outputs * gate_weights, axis=1, keepdims=False)

        if self.enable_gate_weights_metrics:
            gate_weights_batch_mean = tf.reduce_mean(gate_weights, axis=0, keepdims=False)
            for i in range(gate_weights.shape[1]):
                self.add_metric(gate_weights_batch_mean[i], name=f"{self.gate_name}_weight_{i}")

        return out

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape["experts"][0], input_shape["experts"][2]])

    def get_config(self):
        config = super().get_config()
        config.update(
            dim=self.dim,
            num_experts=self.num_experts,
            softmax_temperature=self.softmax_temperature,
            enable_gate_weights_metrics=self.enable_gate_weights_metrics,
        )

        return config


def MMOEBlock(
    input_block: Block,
    outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock],
    expert_block: Block,
    num_experts: int,
    gate_block: Optional[Block] = None,
    gate_softmax_temperature: float = 1.0,
    **gate_kwargs,
) -> SequentialBlock:
    """Implements the Multi-gate Mixture-of-Experts (MMoE) introduced in [1].

    References
    ----------
    [1] Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with
    multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international
    conference on knowledge discovery & data mining. 2018.

    Parameters
    ----------
    input_block : Optional[Block]
        The input block, that will be fed as input for each expert and
        also for the gates, that control the weighted sum of experts
        outputs
    outputs : Union[List[str], List[PredictionTask], ParallelPredictionBlock]
        List with the tasks. A gate is created for each task.
    expert_block : Block
        Expert block to be replicated, e.g. MLPBlock([64])
    num_experts : int
        Number of experts to be replicated
    gate_block : Block, optional
        Allows for having a Block (e.g. MLPBlock([32])) to combine the inputs
        before the final projection layer (created automatically)
        that outputs a softmax distribution over the number of experts.
        This might give more capacity to the gates to decide from the inputs
        how to better combine the experts.
    gate_softmax_temperature : float, optional
        The temperature used by the gates, by default 1.0.
        It can be used to smooth the weights distribution over experts outputs.

    Returns
    -------
    SequentialBlock
        Outputs the sequence of blocks that implement MMOE
    """
    if isinstance(outputs, ParallelPredictionBlock):
        output_names = outputs.task_names
    elif all(isinstance(x, PredictionTask) for x in outputs):
        output_names = [o.task_name for o in outputs]  # type: ignore
    else:
        output_names = outputs  # type: ignore

    experts = expert_block.repeat_in_parallel(
        num_experts, prefix="expert_", aggregation=StackFeatures(axis=1)
    )

    gates = {
        output_name: MMOEGate(
            num_experts,
            gate_block=gate_block.copy() if gate_block else None,
            softmax_temperature=gate_softmax_temperature,
            name=f"gate_{output_name}",
            **gate_kwargs,
        )
        for output_name in output_names
    }
    gates = ParallelBlock(gates)

    mmoe = input_block.connect_with_shortcut(experts, block_outputs_name="experts")
    mmoe = mmoe.connect(gates, block_name="mmoe")

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
        gate_dim: int = 32,
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
            output_names,
            num_task_experts,
            num_shared_experts,
            add_shared_gate=add_shared_gate,
            dim=gate_dim,
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
    """Creates an expert from a block

    Parameters
    ----------
    expert_block : Block
        Expert Block
    name : str
        Expert name

    Returns
    -------
    Tuple[str, TabularBlock]
        Tuple with the expert name and block
    """
    return name, expert_block.as_tabular(name)
