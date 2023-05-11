from typing import Dict, Final, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.utils import module_utils


class TorchScriptWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    accepts_dict: Final[bool]

    def __init__(self, to_wrap: nn.Module, strict: bool = False):
        super().__init__()
        self.to_wrap = to_wrap
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)
        if getattr(to_wrap, "accepts_dict", False):
            self.accepts_dict = True
        else:
            self.accepts_dict = _parallel_check_strict(to_wrap) if strict else False

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    # else-clause is needed to make torchscript happy
                    _batch = batch if batch is not None else Batch({})

                    if not self.accepts_dict:
                        if not torch.jit.isinstance(inputs, torch.Tensor):
                            raise RuntimeError("Expected a tensor, but got a dictionary instead.")
                        x: torch.Tensor = (
                            inputs if isinstance(inputs, torch.Tensor) else inputs["x"]
                        )

                        return self.to_wrap(x, batch=_batch)

                    else:
                        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                            raise RuntimeError("Expected a dictionary, but got a tensor instead.")
                        x: Dict[str, torch.Tensor] = inputs

                        return self.to_wrap(x, batch=_batch)

        if not self.accepts_dict:
            if not torch.jit.isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["x"]

            return self.to_wrap(x)

        else:
            if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                raise RuntimeError("Expected a dictionary, but got a tensor instead.")
            x: Dict[str, torch.Tensor] = inputs

            return self.to_wrap(x)

    def unwrap(self) -> nn.Module:
        return self.to_wrap

    def __getattr__(self, name: str):
        if name in [
            "accepts_dict",
            "accepts_batch",
            "requires_batch",
            "to_wrap",
            "unwrap",
        ]:
            return super().__getattr__(name)

        # Otherwise, return the attribute from the to_wrap module
        return getattr(self.to_wrap, name)

    def __repr__(self):
        return self.to_wrap.__repr__()


def _parallel_check_strict(
    parallel: Dict[str, nn.Module],
    pre: Optional[nn.Module] = None,
    post: Optional[nn.Module] = None,
) -> bool:
    if isinstance(parallel, nn.Module):
        parallel = {"module": parallel}

    pre_input_type, pre_output_type = None, None

    if pre:
        pre_input_type, pre_output_type = module_utils.torchscript_io_types(pre)

    parallel_input_types = {}
    parallel_output_types = {}

    for name, module in parallel.items():
        input_type, output_type = module_utils.torchscript_io_types(module)
        parallel_input_types[name] = input_type
        parallel_output_types[name] = output_type

        if pre and pre_output_type != input_type:
            raise ValueError(
                f"Input type mismatch between pre module and parallel module {name}: {pre_output_type} != {input_type}. "
                "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
                "Consider annotating one of the provided modules."
            )

    first_parallel_input_type = next(iter(parallel_input_types.values()))
    if not all(i_type == first_parallel_input_type for i_type in parallel_input_types.values()):
        raise ValueError(
            f"Input type mismatch among parallel modules: {parallel_input_types}. "
            "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
            "Consider annotating one of the provided modules."
        )

    if post:
        parallel_out = f"Dict[str, {first_parallel_input_type}]"
        post_input_type, _ = module_utils.torchscript_io_types(post)

        if parallel_out != post_input_type:
            raise ValueError(
                f"Output type mismatch between parallel modules and post module: {parallel_output_types} != {post_input_type}. "
                "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
                "Consider annotating one of the provided modules."
            )

    inp_type = pre_input_type if pre else first_parallel_input_type

    return inp_type == "Dict[str, Tensor]"
