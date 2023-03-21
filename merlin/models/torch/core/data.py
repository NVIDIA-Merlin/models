from functools import wraps
from typing import Callable, Dict, Union

import torch
from torch import nn

from merlin.models.torch.utils import module_utils


def _propagate(child: nn.Module, data: Union[Dict[str, torch.Tensor], torch.Tensor], prefix: str):
    if isinstance(data, dict):
        for key, val in data.items():
            prefix = f"{prefix}_{key}"
            _propagate(child, val, prefix)
    else:
        name = f"__buffer_{prefix}"
        if hasattr(child, name):
            setattr(child, name, data)
        else:
            child.register_buffer(name, data, persistent=False)


def _propagate_data(module: nn.Module, features, targets=None):
    for child in module_utils.get_all_children(module):
        _propagate(child, features, "feature")
        _propagate(child, targets, "target")


def propagate_data_to_children(func: Callable):
    @wraps(func)
    def wrapper(module: nn.Module, features, targets=None, *args, **kwargs):
        _propagate_data(module, features, targets)
        return func(module, features, *args, **kwargs)

    return wrapper


def get_features(module: nn.Module):
    raise NotImplementedError


def get_targets(module: nn.Module):
    raise NotImplementedError
