from typing import Dict, Union

import torch
from torch import nn

from merlin.models.torch.utils import module_utils


class DataPropagationHook(nn.Module):
    def __init__(self, propagate_features: bool = True, propagate_targets: bool = True):
        super().__init__()
        self.propagate_features = propagate_features
        self.propagate_targets = propagate_targets

    def forward(self, model, inputs, kwargs):
        targets = kwargs.get("targets", None)
        for child in module_utils.get_all_children(model)[:-1]:
            if self.propagate_features:
                self._upsert_buffers(child, inputs[0], "feature")
            if targets not in (None, {}) and self.propagate_targets:
                self._upsert_buffers(child, targets, "target")

        return inputs, {}

    def _upsert_buffers(
        self, child: nn.Module, data: Union[Dict[str, torch.Tensor], torch.Tensor], prefix: str
    ):
        if isinstance(child, nn.ModuleList):
            for c in child:
                self._upsert_buffers(c, data, prefix)
        elif isinstance(data, dict):
            for key, val in data.items():
                key_prefix = f"{prefix}_{key}"
                self._upsert_buffers(child, val, key_prefix)
        else:
            name = f"__buffer_{prefix}"
            if hasattr(child, name):
                setattr(child, name, data)
            else:
                child.register_buffer(name, data, persistent=False)


def get_features(module: nn.Module) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    prefix = "__buffer_feature"
    features = {}

    for name, buffer in module.named_buffers():
        if name.startswith(prefix):
            features[name] = buffer

    if len(features) == 1:
        return list(features.values)[0]

    return {k[len(prefix) + 1 :]: v for k, v in features.items()}


def get_targets(module: nn.Module) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    prefix = "__buffer_target"
    targets = {}

    for name, buffer in module.named_buffers():
        if name.startswith(prefix):
            targets[name] = buffer

    if len(targets) == 1:
        return list(targets.values)[0]

    return {k[len(prefix) + 1 :]: v for k, v in targets.items()}
