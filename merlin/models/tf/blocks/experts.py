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
from merlin.models.tf.core.combinators import (
    ParallelBlock,
    SequentialBlock,
    TabularBlock,
    WithShortcut,
)
from merlin.models.tf.models.base import get_task_names_from_outputs
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ExpertsGate(Block):
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

    @classmethod
    def from_config(cls, config, **kwargs):
        config = maybe_deserialize_keras_objects(config, ["gate_block"])
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, ["gate_block"])
        config.update(
            num_experts=self.num_experts,
            softmax_temperature=self.softmax_temperature,
            enable_gate_weights_metrics=self.enable_gate_weights_metrics,
            name=self.gate_name,
        )

        return config


def MMOEBlock(
    outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock],
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
    outputs : Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock]
        Names of the tasks or PredictionTask/ParallelPredictionBlock objects from
        which we can extract the task names. A gate is created for each task.
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
    if isinstance(outputs, (tuple, list)) and isinstance(outputs[0], str):
        output_names = outputs  # type: ignore
    else:
        output_names = get_task_names_from_outputs(outputs)

    if not isinstance(expert_block, Block):
        expert_block = Block.from_layer(expert_block)

    experts = expert_block.repeat_in_parallel(
        num_experts, prefix="expert_", aggregation=StackFeatures(axis=1)
    )

    gates = {
        output_name: ExpertsGate(
            num_experts,
            gate_block=gate_block.copy() if gate_block else None,
            softmax_temperature=gate_softmax_temperature,
            name=f"gate_{output_name}",
            **gate_kwargs,
        )
        for output_name in output_names
    }
    gates = ParallelBlock(gates)

    mmoe = WithShortcut(
        experts,
        block_outputs_name="experts",
        automatic_pruning=False,
    )
    mmoe = mmoe.connect(gates, block_name="mmoe")

    return mmoe


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class CGCGateTransformation(TabularBlock):
    """Use the tasks (and shared) gates to aggregate
    the outputs of the experts.

    Parameters
    ----------
    task_names : List[str]
        List with the task names
    num_task_experts : int, optional
        Number of task-specific experts, by default 1
    num_task_experts : int, optional
        Number of task-specific experts, by default 1
    num_shared_experts : int, optional
        Number of shared experts for tasks, by default 1
    add_shared_gate : bool, optional
        Whether to add a shared gate for this CGC block, by default False.
        Useful when multiple CGC blocks are stacked (e.g. in PLEBlock)
        As all CGC blocks except the last one should include the shared gate.
    gate_block : Optional[Block], optional
        Optional block than can make the Gate mode powerful in converting
        the inputs into expert weights for averaging, by default None
    gate_softmax_temperature : float, optional
        Temperature of the softmax used by the Gates for getting
        weights for the average. Temperature can be used to smooth
        the weights distribution, and is by default 1.0.
    enable_gate_weights_metrics : bool, optional
        Enables logging the average gate weights on experts
    gate_dict : Dict[str, ExpertsGate], optional
        Used by the deserializer to rebuild the dict of gates, by default None
    """

    def __init__(
        self,
        task_names: List[str],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        add_shared_gate: bool = False,
        gate_block: Optional[Block] = None,
        gate_softmax_temperature: float = 1.0,
        enable_gate_weights_metrics: bool = False,
        gate_dict: Dict[str, ExpertsGate] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        num_total_experts = num_task_experts + num_shared_experts
        self.task_names = list(task_names)  # Creates another list
        if add_shared_gate and "shared" not in self.task_names:
            self.task_names.append("shared")
        self.stack = StackFeatures(axis=1)

        self.gate_dict = gate_dict
        if self.gate_dict is None:
            self.gate_dict: Dict[str, ExpertsGate] = {
                name: ExpertsGate(
                    num_total_experts,
                    gate_block=gate_block.copy() if gate_block else None,
                    softmax_temperature=gate_softmax_temperature,
                    enable_gate_weights_metrics=enable_gate_weights_metrics,
                    name=f"gate_{name}",
                )
                for name in task_names
            }

            if add_shared_gate:
                self.gate_dict["shared"] = ExpertsGate(
                    (len(task_names) * num_task_experts) + num_shared_experts,
                    gate_block=gate_block,
                    softmax_temperature=gate_softmax_temperature,
                    enable_gate_weights_metrics=enable_gate_weights_metrics,
                    name="shared_gate",
                )

    def call(self, expert_outputs: TabularData, **kwargs) -> TabularData:  # type: ignore
        outputs: TabularData = {}

        expert_outputs = dict(expert_outputs)  # Copying the dict
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

        output_shapes = {name: tensor_output_shape for name in self.task_names}
        return output_shapes

    @classmethod
    def from_config(cls, config, custom_objects=None, **kwargs):
        gate_dict = config.pop("gate_dict")
        if gate_dict is not None:
            gate_dict = {
                name: tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
                for name, conf in gate_dict.items()
            }

        config.update(gate_dict=gate_dict)
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, ["gate_dict"])
        config.update(
            task_names=self.task_names,
        )

        return config


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class CGCBlock(ParallelBlock):
    """Implements the Customized Gate Control (CGC) proposed in [1].

    References
    ----------
    [1] Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task
    learning (mtl) model for personalized recommendations."
    Fourteenth ACM Conference on Recommender Systems. 2020.

    Parameters
    ----------
    outputs : Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock]
       Names of the tasks or PredictionTask/ParallelPredictionBlock objects from
        which we can extract the task names
    expert_block : Union[Block, tf.keras.layers.Layer]
        Block that will be used for the experts
    num_task_experts : int, optional
        Number of task-specific experts, by default 1
    num_shared_experts : int, optional
        Number of shared experts for tasks, by default 1
    add_shared_gate : bool, optional
        Whether to add a shared gate for this CGC block, by default False.
        Useful when multiple CGC blocks are stacked (e.g. in PLEBlock)
        As all CGC blocks except the last one should include the shared gate.
    gate_block : Optional[Block], optional
        Optional block than can make the Gate mode powerful in converting
        the inputs into expert weights for averaging, by default None
    gate_softmax_temperature : float, optional
        Temperature of the softmax used by the Gates for getting
        weights for the average. Temperature can be used to smooth
        the weights distribution, and is by default 1.0.
    enable_gate_weights_metrics : bool, optional
        Enables logging the average gate weights on experts
    name : Optional[str], optional
        Name of the CGC block, by default None
    """

    def __init__(
        self,
        outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock],
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        add_shared_gate: bool = False,
        gate_block: Optional[Block] = None,
        gate_softmax_temperature: float = 1.0,
        enable_gate_weights_metrics: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(expert_block, Block):
            expert_block = Block.from_layer(expert_block)

        if isinstance(outputs, (tuple, list)) and isinstance(outputs[0], str):
            output_names = outputs  # type: ignore
        else:
            output_names = get_task_names_from_outputs(outputs)

        output_names = list(output_names)
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

        experts = {**task_experts, **shared_experts}

        post = kwargs.pop("post", None)
        if post is None:
            post = CGCGateTransformation(
                output_names,
                num_task_experts,
                num_shared_experts,
                add_shared_gate=add_shared_gate,
                gate_block=gate_block,
                gate_softmax_temperature=gate_softmax_temperature,
                enable_gate_weights_metrics=enable_gate_weights_metrics,
            )
        super().__init__(
            experts,
            post=post,
            aggregation=None,
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

    @classmethod
    def from_config(cls, config, custom_objects=None, **kwargs):
        inputs, config = ParallelBlock.parse_config(config, custom_objects)
        paralel_block = ParallelBlock(inputs, **config)
        paralel_block.__class__ = cls
        return paralel_block

    def get_config(self):
        config = super().get_config()
        return config


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
    return name, expert_block.copy().as_tabular(name)


def PLEBlock(
    num_layers: int,
    outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock],
    expert_block: Union[Block, tf.keras.layers.Layer],
    num_task_experts: int = 1,
    num_shared_experts: int = 1,
    gate_block: Optional[Block] = None,
    gate_softmax_temperature: float = 1.0,
    enable_gate_weights_metrics: bool = False,
    name: Optional[str] = None,
    **kwargs,
):
    """Implements the Progressive Layered Extraction (PLE) model from [1],
    by stacking CGC blocks (CGCBlock).

    References
    ----------
    [1] Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task
    learning (mtl) model for personalized recommendations."
    Fourteenth ACM Conference on Recommender Systems. 2020.

    Parameters
    ----------
    num_layers : int
        Number of stacked CGC blocks
    outputs : Union[List[str], List[PredictionTask], ParallelPredictionBlock, ParallelBlock]
        Names of the tasks or PredictionTask/ParallelPredictionBlock objects from
        which we can extract the task names
    expert_block : Union[Block, tf.keras.layers.Layer]
        Block that will be used for the experts
    num_task_experts : int, optional
        Number of task-specific experts, by default 1
    num_shared_experts : int, optional
        Number of shared experts for tasks, by default 1
    gate_block : Optional[Block], optional
        Optional block than can make the Gate mode powerful in converting
        the inputs into expert weights for averaging, by default None
    gate_softmax_temperature : float, optional
        Temperature of the softmax used by the Gates for getting
        weights for the average. Temperature can be used to smooth
        the weights distribution, and is by default 1.0.
    enable_gate_weights_metrics : bool, optional
        Enables logging the average gate weights on experts
    name : Optional[str], optional
        Name of the PLE block, by default None

    Returns
    -------
    SequentialBlock
        Returns the PLE block
    """
    cgc_blocks = []

    for i in range(num_layers):
        cgc_block = CGCBlock(
            outputs=outputs,
            expert_block=expert_block,
            num_task_experts=num_task_experts,
            num_shared_experts=num_shared_experts,
            add_shared_gate=(i < num_layers - 1),
            gate_block=gate_block,
            gate_softmax_temperature=gate_softmax_temperature,
            enable_gate_weights_metrics=enable_gate_weights_metrics,
            name=f"cgc_block_{i}",
            **kwargs,
        )
        cgc_blocks.append(cgc_block)

    cgc_blocks = SequentialBlock(*cgc_blocks, name=name)
    return cgc_blocks
