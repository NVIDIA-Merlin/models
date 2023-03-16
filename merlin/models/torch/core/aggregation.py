from typing import Dict, Union

import torch
from torch import nn


class SumResidual(nn.Module):
    def __init__(self, activation="relu", shortcut_name="shortcut"):
        super().__init__()
        self.activation = getattr(torch, activation) if activation else None
        self.shortcut_name = shortcut_name

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        shortcut = inputs.pop(self.shortcut_name)
        outputs = {}
        for key, val in inputs.items():
            outputs[key] = torch.sum(torch.stack([inputs[key], shortcut]), dim=0)
            if self.activation:
                outputs[key] = self.activation(outputs[key])

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs
