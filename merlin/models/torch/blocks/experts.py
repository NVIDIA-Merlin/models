from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import (
    Block,
    ParallelBlock,
    ShortcutBlock,
    repeat_parallel,
    repeat_parallel_like,
)
from merlin.models.torch.transforms.agg import Stack
from merlin.models.utils.doc_utils import docstring_parameter

_PLE_REFERENCE = """
    References
    ----------
    .. [1] Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task
    learning (mtl) model for personalized recommendations."
    Fourteenth ACM Conference on Recommender Systems. 2020.
"""


class MMOEBlock(Block):
    """
    Multi-gate Mixture-of-Experts (MMoE) Block introduced in [1].

    References
    ----------
    [1] Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with
    multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international
    conference on knowledge discovery & data mining. 2018.

    Parameters
    ----------
    expert : nn.Module
        The base expert model to be used.
    num_experts : int
        The number of experts to be used.
    outputs : Optional[ParallelBlock]
        The output block. If it is an instance of ParallelBlock,
        repeat it for each expert, otherwise use a single ExpertGateBlock.
    """

    def __init__(
        self, expert: nn.Module, num_experts: int, outputs: Optional[ParallelBlock] = None
    ):
        experts = repeat_parallel(expert, num_experts, agg=Stack(dim=1))
        super().__init__(ShortcutBlock(experts, output_name="experts"))
        if isinstance(outputs, ParallelBlock):
            self.append(repeat_parallel_like(ExpertGateBlock(len(outputs)), outputs))
        else:
            self.append(ExpertGateBlock(1))


@docstring_parameter(ple_reference=_PLE_REFERENCE)
class PLEBlock(Block):
    """
    Progressive Layered Extraction (PLE) Block  proposed in [1].

    {ple_reference}

    Parameters
    ----------
    expert : nn.Module
        The base expert model to be used.
    num_shared_experts : int
        The number of shared experts.
    num_task_experts : int
        The number of task-specific experts.
    depth : int
        The depth of the network.
    outputs : ParallelBlock
        The output block.
    """

    def __init__(
        self,
        expert: nn.Module,
        num_shared_experts: int,
        num_task_experts: int,
        depth: int,
        outputs: ParallelBlock,
    ):
        cgc_kwargs = {
            "expert": expert,
            "num_shared_experts": num_shared_experts,
            "num_task_experts": num_task_experts,
            "outputs": outputs,
        }
        super().__init__(*CGCBlock(shared_gate=True, **cgc_kwargs).repeat(depth - 1))
        self.append(CGCBlock(**cgc_kwargs))


class CGCBlock(Block):
    """
    Implements the Customized Gate Control (CGC) proposed in [1].

    {ple_reference}

    Parameters
    ----------
    expert : nn.Module
        The base expert model to be used.
    num_shared_experts : int
        The number of shared experts.
    num_task_experts : int
        The number of task-specific experts.
    outputs : ParallelBlock
        The output block.
    shared_gate : bool, optional
        If true, use a shared gate for all tasks. Defaults to False.
    """

    def __init__(
        self,
        expert: nn.Module,
        num_shared_experts: int,
        num_task_experts: int,
        outputs: ParallelBlock,
        shared_gate: bool = False,
    ):
        shared_experts = repeat_parallel(expert, num_shared_experts, agg="stack")
        super().__init__(ShortcutBlock(shared_experts, output_name="experts"))

        gates = ParallelBlock()
        for name in outputs.branches:
            gates[name] = PLEExpertGateBlock(
                len(outputs),
                experts=repeat_parallel(expert, num_task_experts, agg="stack"),
                name=name,
            )
        if shared_gate:
            gates["experts"] = ExpertGateBlock(len(outputs))

        self.append(gates)


class ExpertGateBlock(Block):
    """Expert Gate Block.

    Parameters
    ----------
    num_outputs : int
        The number of output channels.
    """

    def __init__(self, num_outputs: int):
        super().__init__(GateBlock(num_outputs))

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        if torch.jit.isinstance(inputs, torch.Tensor):
            raise RuntimeError("ExpertGateBlock requires a dictionary input")

        experts = inputs["experts"]
        outputs = inputs["shortcut"]
        for module in self.values:
            outputs = module(outputs, batch=batch)

        gated = outputs.expand_as(experts)

        # Multiply and sum along the experts dimension
        return (experts * gated).sum(dim=1)


class PLEExpertGateBlock(Block):
    """
    Progressive Layered Extraction (PLE) Expert Gate Block.

    Parameters
    ----------
    num_outputs : int
        The number of output channels.
    experts : nn.Module
        The expert module.
    name : str
        The name of the task.
    """

    def __init__(self, num_outputs: int, experts: nn.Module, name: str):
        super().__init__(GateBlock(num_outputs), name=f"PLEExpertGateBlock[{name}]")
        self.stack = Stack(dim=1)
        self.experts = experts
        self.task_name = name

    def forward(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        task_experts = self.experts(inputs["shortcut"], batch=batch)
        experts = self.stack({"experts": inputs["experts"], "task_experts": task_experts})
        task = inputs[self.name] if self.name in inputs else inputs["shortcut"]

        return self.output({"experts": experts, "shortcut": task})


class SoftmaxGate(nn.Module):
    """Softmax Gate for gating mechanism."""

    def forward(self, gate_logits):
        return torch.softmax(gate_logits, dim=1).unsqueeze(2)


class GateBlock(Block):
    """Gate Block for gating mechanism."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.append(nn.LazyLinear(num_outputs))
        self.append(SoftmaxGate())
