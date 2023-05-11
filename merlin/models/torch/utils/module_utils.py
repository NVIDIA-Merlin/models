import inspect
from typing import Dict, Tuple

import torch
import torch.nn as nn


def is_tabular(module: torch.nn.Module) -> bool:
    # Get the forward method of the input module
    forward_method = module.forward

    # Get the signature of the forward method
    forward_signature = inspect.signature(forward_method)

    # Get the first argument of the forward method
    first_arg = list(forward_signature.parameters.values())[0]

    # Check if the annotation exists for the first argument
    if first_arg.annotation != inspect.Parameter.empty:
        # Check if the annotation is a dict of tensors
        if first_arg.annotation == Dict[str, torch.Tensor]:
            return True

    return False


def check_batch_arg(module: nn.Module) -> Tuple[bool, bool]:
    accepts_batch = False
    requires_batch = False

    forward_signature = inspect.signature(module.forward)
    num_args = len(forward_signature.parameters)
    accepts_batch = "batch" in forward_signature.parameters

    if accepts_batch:
        batch_arg = forward_signature.parameters["batch"]
        requires_batch = batch_arg.default is not None

    if accepts_batch and num_args > 1:
        return accepts_batch, requires_batch

    return False, False


def module_test(module, input_data, method="script", **kwargs):
    from merlin.models.torch.batch import Batch

    # Check if the module can be called with the provided inputs
    try:
        original_output = module(input_data, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to call the module with provided inputs: {e}")

    # Check if the module can be scripted
    try:
        if method == "script":
            scripted_module = torch.jit.script(module)
        elif method == "trace":
            scripted_module = torch.jit.trace(module, input_data, strict=True)
        else:
            raise ValueError(f"Unknown method: {method}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to script the module: {e}")

    # Compare the output of the original module and the scripted module
    with torch.no_grad():
        scripted_output = scripted_module(input_data, **kwargs)

    if isinstance(original_output, dict):
        _all_close_dict(original_output, scripted_output)
    elif isinstance(original_output, tuple):
        for i in range(len(original_output)):
            if not torch.allclose(original_output[i], scripted_output[i]):
                raise ValueError(
                    "The outputs of the original and scripted modules are not the same"
                )
    elif isinstance(original_output, Batch):
        _all_close_dict(original_output.features, scripted_output.features)
        if original_output.targets is not None:
            _all_close_dict(original_output.targets, scripted_output.targets)
        if original_output.sequences is not None:
            _all_close_dict(
                original_output.sequences.seq_lengths, scripted_output.sequences.seq_lengths
            )
            if original_output.sequences.masks is not None:
                _all_close_dict(original_output.sequences.masks, scripted_output.sequences.masks)
    else:
        if not torch.allclose(original_output, scripted_output):
            raise ValueError("The outputs of the original and scripted modules are not the same")

    return original_output


def _all_close_dict(left, right):
    for key in left.keys():
        if not torch.allclose(left[key], right[key]):
            raise ValueError("The outputs of the original and scripted modules are not the same")
