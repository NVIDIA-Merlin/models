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


class MMOEBlock(Block):
    def __init__(
        self, expert: nn.Module, num_experts: int, outputs: Optional[ParallelBlock] = None
    ):
        experts = repeat_parallel(expert, num_experts, agg=Stack(dim=1))
        super().__init__(ShortcutBlock(experts, output_name="experts"))
        if isinstance(outputs, ParallelBlock):
            self.append(repeat_parallel_like(ExpertGateBlock(len(outputs)), outputs))
        else:
            self.append(ExpertGateBlock(1))


class PLEBlock(Block):
    def __init__(
        self,
        expert: nn.Module,
        num_shared_experts: int,
        num_task_experts: int,
        depth: int,
        outputs: ParallelBlock,
    ):
        cgc = CGCBlock(
            expert, num_shared_experts, num_task_experts, outputs=outputs, shared_gate=True
        )
        super().__init__(*cgc.repeat(depth - 1))
        self.append(CGCBlock(expert, num_shared_experts, num_task_experts, outputs=outputs))


class CGCBlock(Block):
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
    def forward(self, gate_logits):
        return torch.softmax(gate_logits, dim=1).unsqueeze(2)


class GateBlock(Block):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.append(nn.LazyLinear(num_outputs))
        self.append(SoftmaxGate())
