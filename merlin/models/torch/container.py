#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import builtins
from copy import deepcopy
from functools import reduce, wraps
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from merlin.models.torch.utils import torchscript_utils


@runtime_checkable
class HasBool(Protocol):
    def __bool__(self) -> bool:
        ...


_TModule = TypeVar("_TModule", bound=nn.Module)
ModuleFunc = Callable[[nn.Module], nn.Module]
ModuleIFunc = Callable[[nn.Module, int], nn.Module]
ModulePredicate = Callable[[nn.Module], Union[bool, HasBool]]


class ContainerMixin:
    def filter(self: _TModule, func: ModulePredicate, recurse: bool = False) -> _TModule:
        _to_call = _recurse(func, "filter") if recurse else func
        output = self.__class__()

        for module in self:
            if _to_call(module):
                output.append(module)

        return output

    def flatmap(self: _TModule, func: ModuleFunc) -> _TModule:
        if not callable(func):
            raise TypeError(f"Expected callable function, received: {type(func).__name__}")

        try:
            mapped = self.map(func)
        except Exception as e:
            raise RuntimeError("Failed to map function over the module") from e

        output = self.__class__()

        try:
            for sublist in mapped:
                for item in sublist:
                    output.append(item)
        except TypeError as e:
            raise TypeError("Function did not return an iterable object") from e

        return output

    def forall(self, func: ModulePredicate, recurse: bool = False) -> bool:
        _to_call = _recurse(func, "forall") if recurse else func
        return all(_to_call(module) for module in self)

    def map(self: _TModule, func: ModuleFunc, recurse: bool = False) -> _TModule:
        _to_call = _recurse(func, "map") if recurse else func

        return self.__class__(*(_to_call(module) for module in self))

    def mapi(self: _TModule, func: ModuleIFunc, recurse: bool = False) -> _TModule:
        _to_call = _recurse(func, "mapi") if recurse else func
        return self.__class__(*(_to_call(module, i) for i, module in enumerate(self)))

    def choose(self: _TModule, func: ModulePredicate, recurse: bool = False) -> _TModule:
        to_add = []
        _to_call = _recurse(func, "choose") if recurse else func

        for module in self:
            f_out = _to_call(module)
            if f_out:
                to_add.append(f_out)

        return self.__class__(*to_add)

    def walk(self: _TModule, func: ModulePredicate) -> _TModule:
        return self.map(func, recurse=True)

    def zip(self, other: Iterable[_TModule]) -> Iterable[Tuple[_TModule, _TModule]]:
        return builtins.zip(self, other)

    def __add__(self, module) -> _TModule:
        return self.__class__(*self, module)

    def __radd__(self, module) -> _TModule:
        return self.__class__(module, *self)


class BlockContainer(nn.Module, Iterable[_TModule], ContainerMixin):
    """A container class for PyTorch `nn.Module` that allows for manipulation and traversal
    of multiple sub-modules as if they were a list. The modules are automatically wrapped
    in a TorchScriptWrapper for TorchScript compatibility.

    Parameters
    ----------
    *inputs : nn.Module
        One or more PyTorch modules to be added to the container.
    name : Optional[str]
        An optional name for the BlockContainer.
    """

    def __init__(self, *inputs: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.values = nn.ModuleList()

        for module in inputs:
            self.values.append(self.wrap_module(module))

        self._name: str = name

    def append(self, module: nn.Module):
        """Appends a given module to the end of the list.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be appended.

        Returns
        -------
        self
        """
        self.values.append(self.wrap_module(module))

        return self

    def extend(self, sequence: Sequence[nn.Module]):
        """Extends the list by appending elements from the iterable.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be appended.

        Returns
        -------
        self
        """
        for m in sequence:
            self.append(m)

        return self

    def prepend(self, module: nn.Module):
        """Prepends a given module to the beginning of the list.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be prepended.

        Returns
        -------
        self
        """
        return self.insert(0, module)

    def insert(self, index: int, module: nn.Module):
        """Inserts a given module at the specified index.

        Parameters
        ----------
        index : int
            The index at which the module is to be inserted.
        module : nn.Module
            The PyTorch module to be inserted.

        Returns
        -------
        self
        """
        self.values.insert(index, self.wrap_module(module))

        return self

    def unwrap(self) -> nn.ModuleList:
        return self

    def wrap_module(
        self, module: nn.Module
    ) -> Union["BlockContainer", torchscript_utils.TorchScriptWrapper]:
        if isinstance(module, (BlockContainer, torchscript_utils.TorchScriptWrapper)):
            return module

        return torchscript_utils.TorchScriptWrapper(module)

    def add_module(
        self,
        name: str,
        module: Optional[nn.Module],
    ) -> None:
        _module = torchscript_utils.TorchScriptWrapper(module)

        super().add_module(name, _module)

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self.values)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(m.unwrap() for m in self.values)

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            return BlockContainer(*[v for v in self.values[idx]])

        return self.values[idx].unwrap()

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        if not isinstance(module, torchscript_utils.TorchScriptWrapper):
            module = torchscript_utils.TorchScriptWrapper(module)

        self.values[idx] = module

        return self

    def __delitem__(self, idx: Union[slice, int]) -> None:
        self.values.__delitem__(idx)

    def __add__(self, other) -> "BlockContainer":
        for module in other:
            self.append(module)

        return self

    def __bool__(self) -> bool:
        return bool(self.values)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BlockContainer):
            return False

        if len(self) != len(other):
            return False

        return all(a == b for a, b in zip(self, other))

    def __hash__(self) -> int:
        return hash(tuple(self.values))

    def __repr__(self) -> str:
        sequential = repr(self.values)

        return self._get_name() + sequential[len("ModuleList") :]

    def _get_name(self) -> str:
        return super()._get_name() if self._name is None else self._name


class BlockContainerDict(nn.ModuleDict):
    """A container class for PyTorch `nn.Module` that allows for manipulation and traversal
    of multiple sub-modules as if they were a dictionary. The modules are automatically wrapped
    in a TorchScriptWrapper for TorchScript compatibility.

    Parameters
    ----------
    *inputs : nn.Module
        One or more PyTorch modules to be added to the container.
    name : Optional[str]
        An optional name for the BlockContainer.
    """

    def __init__(
        self,
        *inputs: Union[nn.Module, Dict[str, nn.Module]],
        name: Optional[str] = None,
        block_cls=BlockContainer,
    ) -> None:
        if not inputs:
            inputs = [{}]

        if all(isinstance(x, dict) for x in inputs):
            modules = reduce(lambda a, b: dict(a, **b), inputs)  # type: ignore

        self._block_cls = block_cls
        super().__init__(modules)
        self._name: str = name

    def append_to(self, name: str, module: nn.Module) -> "BlockContainerDict":
        """Appends a module to a specified name.

        Parameters
        ----------
        name : str
            The name of the branch.
        module : nn.Module
            The module to append.

        Returns
        -------
        BlockContainerDict
            The current object itself.
        """

        self._modules[name].append(module)

        return self

    def prepend_to(self, name: str, module: nn.Module) -> "BlockContainerDict":
        """Prepends a module to a specified name.

        Parameters
        ----------
        name : str
            The name of the branch.
        module : nn.Module
            The module to prepend.

        Returns
        -------
        BlockContainerDict
            The current object itself.
        """

        self._modules[name].prepend(module)

    def append_for_each(self, module: nn.Module, shared=False) -> "BlockContainerDict":
        """Appends a module to each branch.

        Parameters
        ----------
        module : nn.Module
            The module to append to each branch.
        shared : bool, default=False
            If True, the same module is shared across all elements.
            Otherwise a deep copy of the module is used in each element.

        Returns
        -------
        BlockContainerDict
            The current object itself.
        """

        for branch in self.values():
            _module = module if shared else deepcopy(module)
            branch.append(_module)

        return self

    def prepend_for_each(self, module: nn.Module, shared=False) -> "BlockContainerDict":
        """Prepends a module to each branch.

        Parameters
        ----------
        module : nn.Module
            The module to prepend to each branch.
        shared : bool, default=False
            If True, the same module is shared across all elements.
            Otherwise a deep copy of the module is used in each element.

        Returns
        -------
        BlockContainerDict
            The current object itself.
        """
        for branch in self.values():
            _module = module if shared else deepcopy(module)
            branch.prepend(_module)

        return self

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        if module and not isinstance(module, self._block_cls):
            module = self._block_cls(module)

        output = super().add_module(name, module)

        return output

    def unwrap(self) -> Dict[str, nn.ModuleList]:
        return {k: v.unwrap() for k, v in self.items()}

    def _get_name(self) -> str:
        return super()._get_name() if self._name is None else self._name

    def __eq__(self, other: "BlockContainerDict") -> bool:
        if not isinstance(other, BlockContainerDict):
            return False

        return all(other[key] == val if key in other else False for key, val in self.items())

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._modules.items())))


def _recurse(func, to_recurse_name: str):
    @wraps(func)
    def inner(module, **kwargs):
        if hasattr(module, to_recurse_name):
            fn = getattr(module, to_recurse_name)

            return fn(func, **kwargs)

        return func(module, **kwargs)

    return inner
