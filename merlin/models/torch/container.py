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

from copy import deepcopy
from functools import reduce
from typing import Dict, Iterator, Optional, Union

from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from merlin.models.torch.utils import torchscript_utils


class BlockContainer(nn.Module):
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
        return nn.ModuleList([m.unwrap() for m in self])

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
        return iter(self.values)

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]):
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
        self, *inputs: Union[nn.Module, Dict[str, nn.Module]], name: Optional[str] = None
    ) -> None:
        if not inputs:
            inputs = [{}]

        if all(isinstance(x, dict) for x in inputs):
            modules = reduce(lambda a, b: dict(a, **b), inputs)  # type: ignore

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

        return self

    # Append to all branches, optionally copying
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

    # This assumes same branches, we append it's content to each branch
    # def append_parallel(self, module: "BlockContainerDict") -> "BlockContainerDict":
    #     ...

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        if module and not isinstance(module, BlockContainer):
            module = BlockContainer(module, name=name[0].upper() + name[1:])

        return super().add_module(name, module)

    def unwrap(self) -> Dict[str, nn.ModuleList]:
        return {k: v.unwrap() for k, v in self.items()}

    def _get_name(self) -> str:
        return super()._get_name() if self._name is None else self._name
