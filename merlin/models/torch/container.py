from functools import reduce
from typing import Dict, Final, Iterator, Optional, Tuple, Union

import torch
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from merlin.models.torch.data import Batch
from merlin.models.torch.utils import module_utils


class Parallel(nn.Module):
    accepts_dict: Final[bool]

    def __init__(
        self,
        *inputs: Union[nn.Module, Dict[str, nn.Module]],
        strict: bool = True,
    ):
        super().__init__()

        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]

        if all(isinstance(x, dict) for x in inputs):
            _parallel_dict = reduce(lambda a, b: dict(a, **b), inputs)
        elif all(isinstance(x, nn.Module) for x in inputs):
            if all(hasattr(m, "name") for m in inputs):
                _parallel_dict = {m.name: m for m in inputs}
            else:
                _parallel_dict = {i: m for i, m in enumerate(inputs)}
        else:
            raise ValueError(f"Invalid input. Got: {inputs}")

        if not strict:
            self.accepts_dict = False
        else:
            # TODO: Handle with pre
            self.accepts_dict = _parallel_check_strict(_parallel_dict)

        self.branches = _WrappedModuleDict({str(i): m for i, m in _parallel_dict.items()})

    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Optional[Batch] = None,
    ):
        return self.forward_branches(inputs, batch=batch)

    def forward_branches(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Optional[Batch] = None,
    ):
        """
        Process inputs through the parallel layers.

        Parameters
        ----------
        inputs : Tensor
            Input tensor to process through the parallel layers.
        **kwargs : dict
            Additional keyword arguments for layer processing.

        Returns
        -------
        outputs : dict
            Dictionary containing the outputs of the parallel layers.
        """
        if not self.accepts_dict:
            if not torch.jit.isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["x"]

            outputs = {}

            for name, module in self.branches.items():
                module_inputs = x  # TODO: Add filtering when adding schema
                out = module(module_inputs, batch=batch)

                if isinstance(out, torch.Tensor):
                    out = {name: out}
                elif isinstance(out, tuple):
                    out = {name: out}

                for key in out.keys():
                    if key in outputs:
                        raise RuntimeError(
                            f"Duplicate keys found in outputs. "
                            f"Got: {list(out.keys())} and {list(outputs.keys())}"
                        )
                outputs.update(out)

            return outputs

        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            raise RuntimeError("Expected a dictionary, but got a tensor instead.")
        x: Dict[str, torch.Tensor] = inputs

        outputs = {}

        for name, module in self.branches.items():
            module_inputs = x  # TODO: Add filtering when adding schema
            out = module(module_inputs, batch=batch)

            if isinstance(out, torch.Tensor):
                out = {name: out}
            elif isinstance(out, tuple):
                out = {name: out}

            for key in out.keys():
                if key in outputs:
                    raise RuntimeError(
                        f"Duplicate keys found in outputs. "
                        f"Got: {list(out.keys())} and {list(outputs.keys())}"
                    )
            outputs.update(out)

        return outputs

    @_copy_to_script_wrapper
    def items(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.branches.items()

    @_copy_to_script_wrapper
    def keys(self) -> Iterator[str]:
        return self.branches.keys()

    @_copy_to_script_wrapper
    def values(self) -> Iterator[nn.Module]:
        return self.branches.values()

    # @torch.jit.export
    def _first(self):
        for b in self.branches.values():
            return b

    # @property
    @torch.jit.ignore
    def first(self):
        return self._first()

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self.branches)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.branches.values())

    @_copy_to_script_wrapper
    def __getitem__(self, key) -> nn.Module:
        return self.branches[key]
    
    def __setitem__(self, key: str, module: nn.Module) -> None:
        self.branches.add_module(key, module)

    def __bool__(self) -> bool:
        return bool(self.branches)


class WithShortcut(Parallel):
    def __init__(
        self,
        module: nn.Module,
        strict: bool = True,
        module_output_name="output",
        shortcut_output_name="shortcut",
    ):
        shortcut = nn.Identity()
        if strict:
            accepts_dict = _parallel_check_strict({"module": module})
            if accepts_dict:
                shortcut = TabularIdentity()

        branches = {module_output_name: module, shortcut_output_name: shortcut}

        super().__init__(branches, strict=strict)


class TabularIdentity(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return x


class _WrappedModuleList(nn.ModuleList):
    def add_module(self, name: str, module: Optional[nn.Module]):
        if module is not None:
            module = _Wrapper(module)

        return super().add_module(name, module)

    def _get_name(self):
        return "ModuleList"
    
    def unwrap(self) -> nn.ModuleList:
        return nn.ModuleList([m.unwrap() for m in self])

class _WrappedModuleDict(nn.ModuleDict):
    def add_module(self, name: str, module: Optional[nn.Module]):
        if module is not None:
            module = _Wrapper(module)

        return super().add_module(name, module)

    def _get_name(self):
        return "ModuleDict"
    
    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> nn.Module:
        return self._modules[key].unwrap()


class _WrappedTabularModuleDict(nn.ModuleDict):
    def add_module(self, name: str, module: Optional[nn.Module]):
        if module is not None:
            module = _WrapperTabular(module)

        return super().add_module(name, module)

    def _get_name(self):
        return "ModuleDict"


class _Wrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    accepts_dict: Final[bool]

    def __init__(self, to_wrap: nn.Module, name: Optional[str] = None, strict: bool = True):
        super().__init__()
        self.to_wrap = to_wrap
        self.to_wrap_name: str = name or to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
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

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _WrapperTabular(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]

    def __init__(self, to_wrap: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.to_wrap = to_wrap
        self.to_wrap_name: str = name or to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)

    def forward(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
    ) -> Dict[str, torch.Tensor]:
        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    # else-clause is needed to make torchscript happy
                    _batch = batch if batch is not None else Batch({})

                    return self.to_wrap(inputs, batch=_batch)

        return self.to_wrap(inputs)

    def unwrap(self) -> nn.Module:
        return self.to_wrap

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


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
