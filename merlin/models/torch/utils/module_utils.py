import uuid
import re
from inspect import getfullargspec
from typing import Dict, List, Sequence, Type, TypeVar, Union, Tuple

import torch
import torch.nn as nn

from merlin.models.utils.misc_utils import filter_kwargs


@torch.jit.script
def to_tuple(x):
    if not torch.jit.isinstance(x, Tuple[torch.Tensor]):
        return (x, )
    
    return x


def _extract_output_type(module: nn.Module) -> str:
    compiled = torch.jit.script(module)
    
    # Compile a regular expression pattern to match the output 
    # type annotation in a function signature
    # The pattern consists of a main type name followed by an optional 
    # type information inside square brackets
    pattern = re.compile(r"->\s+((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\[.*\])?)")
    match = pattern.search(compiled.code)
    if match:
        return match.group(1)
    else:
        return "Any"
    
    
def _extract_types(module: nn.Module) -> Tuple[str, str]:
    compiled = torch.jit.script(module)

    # Compile regular expression patterns to match the input and output
    # type annotations in a function signature
    input_pattern = re.compile(r"def .+?\(.*?self,\s*?[a-zA-Z_][a-zA-Z0-9_]*:\s+((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\[.*\])?)")
    output_pattern = re.compile(r"->\s+((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\[.*\])?)")

    # Search for input type annotation
    input_match = input_pattern.search(compiled.code)
    if input_match:
        input_type = input_match.group(1)
    else:
        input_type = "Any"

    # Search for output type annotation
    output_match = output_pattern.search(compiled.code)
    if output_match:
        output_type = output_match.group(1)
    else:
        output_type = "Any"

    return input_type, output_type


def ModulePreHook(module: nn.Module):    
    inp_type, out_type = _extract_types(module)
    
    def _hook_t_to_t(
        parent, 
        inputs: Tuple[torch.Tensor], 
    ) -> torch.Tensor:
        return module(*inputs)
    
    def _hook_dict_to_t(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]], 
    ) -> torch.Tensor:
        return module(*inputs)
        
    def _hook_t_to_tuple(
        parent, 
        inputs: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        return module(*inputs)
    
    def _hook_dict_to_tuple(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor]:
        return module(*inputs)
        
    def _hook_t_to_dict(
        parent, 
        inputs: Tuple[torch.Tensor], 
    ) -> Dict[str, torch.Tensor]:
        return module(*inputs)
    
    def _hook_dict_to_dict(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]], 
    ) -> Dict[str, torch.Tensor]:
        return module(*inputs)
    
    if inp_type == "Tensor" and out_type == "Tensor":
        hook = _hook_t_to_t
    elif inp_type == "Dict[str, Tensor]" and out_type == "Tensor":
        hook = _hook_dict_to_t
    elif inp_type == "Tensor" and out_type == "Tuple[Tensor]":
        hook = _hook_t_to_tuple
    elif inp_type == "Dict[str, Tensor]" and out_type == "Tuple[Tensor]":
        hook = _hook_dict_to_tuple
    elif inp_type == "Tensor" and out_type == "Dict[str, Tensor]":
        hook = _hook_t_to_dict
    elif inp_type == "Dict[str, Tensor]" and out_type == "Dict[str, Tensor]":
        hook = _hook_dict_to_dict
    else:
        raise ValueError(
            f"Unsupported output type: {out_type}"
            " for module: {module._get_name()}."
            " Supported output types are: torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]."
            " Please annotate the return type of the forward function of the module."
        )
        
    return hook


def ModulePostHook(module: nn.Module):    
    inp_type, out_type = _extract_types(module)
    
    def _hook_t_to_t(
        parent, 
        inputs: Tuple[torch.Tensor], 
        outputs
    ) -> torch.Tensor:
        return module(outputs)
    
    def _hook_dict_to_t(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]], 
        outputs
    ) -> torch.Tensor:
        return module(outputs)
        
    def _hook_t_to_tuple(
        parent, 
        inputs: Tuple[torch.Tensor],
        outputs
    ) -> Tuple[torch.Tensor]:
        return module(outputs)
    
    def _hook_dict_to_tuple(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]], 
        outputs
    ) -> Tuple[torch.Tensor]:
        return module(outputs)
        
    def _hook_t_to_dict(
        parent, 
        inputs: Tuple[torch.Tensor], 
        outputs
    ) -> Dict[str, torch.Tensor]:
        return module(outputs)
    
    def _hook_dict_to_dict(
        parent, 
        inputs: Tuple[Dict[str, torch.Tensor]], 
        outputs
    ) -> Dict[str, torch.Tensor]:
        return module(outputs)
    
    if inp_type == "Tensor" and out_type == "Tensor":
        hook = _hook_t_to_t
    elif inp_type == "Dict[str, Tensor]" and out_type == "Tensor":
        hook = _hook_dict_to_t
    elif inp_type == "Tensor" and out_type == "Tuple[Tensor]":
        hook = _hook_t_to_tuple
    elif inp_type == "Dict[str, Tensor]" and out_type == "Tuple[Tensor]":
        hook = _hook_dict_to_tuple
    elif inp_type == "Tensor" and out_type == "Dict[str, Tensor]":
        hook = _hook_t_to_dict
    elif inp_type == "Dict[str, Tensor]" and out_type == "Dict[str, Tensor]":
        hook = _hook_dict_to_dict
    else:
        raise ValueError(
            f"Unsupported output type: {out_type}"
            " for module: {module._get_name()}."
            " Supported output types are: torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]."
            " Please annotate the return type of the forward function of the module."
        )
    
    hook.__name__ = "_".join([module._get_name(), uuid.uuid4().hex[10:]])
    
    return hook


# class ModuleHook(nn.Module):
#     """A helper module to be able to execute Module as a (pre) forward-hook."""

#     def __init__(self, module: nn.Module):
#         super().__init__()
#         self.module = module

#     def forward(self, parent, inputs, outputs=None):
#         """
#         Forward pass for the ModuleHook.

#         This function is called when the hook is executed during the
#         forward pass of the parent module.

#         Args:
#             parent (nn.Module): The parent module for which the hook is registered.
#             inputs: The inputs to the parent module.
#             outputs (optional): The outputs of the parent module (only provided for post-hooks).

#         Returns:
#             The result of executing the hook module with the given inputs or outputs.
#         """
#         del parent

#         x = inputs if outputs is None else outputs
#         if isinstance(x, tuple):
#             return self.module(*x)

#         return self.module(x)
    
#     @property
#     def __name__(self):
#         return self.module._get_name()
    
#     @property
#     def __globals__(self):
#         return {}


def has_custom_call(module: nn.Module) -> bool:
    module_call = getattr(module, "__call__", None)
    base_call = getattr(nn.Module, "__call__", None)

    if module_call is None or base_call is None:
        return False

    return module_call.__func__ != base_call


def apply(
    module: Union[nn.Module, Sequence[nn.Module]], inputs, *args, model_context=None, **kwargs
):
    """
    Calls a module with the given inputs and filters kwargs. Returns the output.

    Parameters
    ----------
    module : torch.nn.Module or List[torch.nn.Module]
        The PyTorch module or list of modules to call.
    inputs : torch.Tensor
        The input tensor to be passed to the module.
    *args : tuple
        Additional arguments to be passed to the module.
    **kwargs : dict
        Additional keyword arguments to be filtered and passed to the module.

    Returns
    -------
    output : torch.Tensor
        The output tensor after processing the input by the module.
    """

    _k = dict(cascade_kwargs_if_possible=True, argspec_fn=getfullargspec)

    if isinstance(module, (list, tuple, nn.ModuleList)):
        output = inputs
        for i, mod in enumerate(module):
            if i == 0:
                output = apply(mod, output, *args, model_context=model_context, **kwargs)
            else:
                output = apply(mod, output, model_context=model_context)

        return output

    filtered_kwargs = filter_kwargs(kwargs, module, **_k)

    if not has_custom_call(module) or getattr(module, "check_forward", False):
        # We need to check the forward method on the type since when the model gets saved
        # we can't infer the kwargs from using `module.forward` directly
        forward_fn = type(module).forward
        filtered_kwargs = filter_kwargs(filtered_kwargs, forward_fn, **_k)

    return module(inputs, *args, **filtered_kwargs)


def get_all_children(module: nn.Module) -> List[nn.Module]:
    children = []
    for child in module.children():
        if not isinstance(child, nn.ModuleList):
            children.append(child)
        children.extend(get_all_children(child))

    return children


ToSearch = TypeVar("ToSearch", bound=Type[nn.Module])


def find_all_instances(module: nn.Module, to_search: ToSearch) -> List[ToSearch]:
    if isinstance(to_search, nn.Module):
        to_search = type(to_search)

    if isinstance(module, to_search):
        return [module]
    elif module == to_search:
        return [module]

    children = module.children()
    if children:
        results = []
        for sub_module in children:
            results.extend(find_all_instances(sub_module, to_search))

        return results

    return []


def _to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != "_":
        return insecure
    return "private" + insecure


def module_name(module: nn.Module, snakecase=True) -> str:
    cls_name = module.__class__.__name__

    if snakecase:
        return _to_snake_case(cls_name)

    return cls_name


def module_test(module, input_data):
    # Check if the module can be called with the provided inputs
    try:
        original_output = module(input_data)
    except Exception as e:
        raise RuntimeError(f"Failed to call the module with provided inputs: {e}")

    # Check if the module can be scripted
    try:
        scripted_module = torch.jit.script(module)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to script the module: {e}")

    # Compare the output of the original module and the scripted module
    with torch.no_grad():
        scripted_output = scripted_module(input_data)
        
    if isinstance(original_output, dict):
        for key in original_output.keys():
            if not torch.allclose(original_output[key], scripted_output[key]):
                raise ValueError("The outputs of the original and scripted modules are not the same")
    elif isinstance(original_output, tuple):
        for i in range(len(original_output)):
            if not torch.allclose(original_output[i], scripted_output[i]):
                raise ValueError("The outputs of the original and scripted modules are not the same")
    else:
        if not torch.allclose(original_output, scripted_output):
            raise ValueError("The outputs of the original and scripted modules are not the same")

    return original_output
