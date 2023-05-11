from copy import deepcopy
from functools import reduce
from typing import Dict, Iterator, Optional, Union

from rich import print as rprint
from rich.table import Table
from rich.tree import Tree
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from merlin.models.torch.link import Link
from merlin.models.torch.utils import rich_utils, torchscript_utils


class BlockContainer(nn.Module):
    def __init__(self, *inputs: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.values = nn.ModuleList()

        for module in inputs:
            _module = self._check_link(module)
            self.values.append(self.wrap_module(_module))
        
        self._name: str = name

    def append(self, module: nn.Module, link: Optional[Union[Link, str]] = None):
        _module = self._check_link(module, link=link)
        self.values.append(self.wrap_module(_module))

        return self

    def prepend(self, module: nn.Module, link: Optional[Union[Link, str]] = None):
        return self.insert(0, module, link=link)

    def insert(self, index: int, module: nn.Module, link: Optional[Union[Link, str]] = None):
        _module = self._check_link(module, link=link)
        self.values.insert(index, self.wrap_module(_module))

        return self

    def unwrap(self) -> nn.ModuleList:
        return nn.ModuleList([m.unwrap() for m in self])

    def wrap_module(
        self, module: nn.Module
    ) -> Union["BlockContainer", torchscript_utils.TorchScriptWrapper]:
        if isinstance(
            module, (BlockContainer, BlockContainerDict, torchscript_utils.TorchScriptWrapper)
        ):
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

    def __rich_repr__(self):
        name = self._get_name()
        extra = self.extra_repr()
        if extra:
            name = f"{name}({extra})"
        tree = Tree(name)
        for module in self.values:
            if module:
                if isinstance(module, torchscript_utils.TorchScriptWrapper):
                    module = module.unwrap()
                if hasattr(module, "__rich_repr__"):
                    tree.add(module.__rich_repr__())
                else:
                    tree.add(repr(module))

        return tree
    
    def _repr_html_(self):
        return self.__rich_repr__()

    def rich_print(self):
        rprint(self.__rich_repr__())

    def _get_name(self) -> str:
        return super()._get_name() if self._name is None else self._name

    def _check_link(self, module: nn.Module, link: Optional[Union[Link, str]] = None) -> nn.Module:
        if link:
            _module = Link.parse(link)
            # TODO: Fix this, since self[-1] could be None
            _module.setup_link(self[-1], module)

            return _module

        return module


class BlockContainerDict(nn.ModuleDict):
    def __init__(
        self, *inputs: Union[nn.Module, Dict[str, nn.Module]], name: Optional[str] = None
    ) -> None:
        if not inputs:
            inputs = [{}]

        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            modules = inputs[0]
        if all(isinstance(x, dict) for x in inputs):
            modules = reduce(lambda a, b: dict(a, **b), inputs)  # type: ignore

        super().__init__(modules)
        self._name: str = name

    def append_to(self, name: str, module: nn.Module, link=None) -> "BlockContainerDict":
        self._modules[name].append(module, link=link)

        return self

    def prepend_to(self, name: str, module: nn.Module, link=None) -> "BlockContainerDict":
        self._modules[name].prepend(module, link=link)

        return self

    # Append to all branches, optionally copying
    def append_for_each(self, module: nn.Module, copy=False, link=None) -> "BlockContainerDict":
        for branch in self.values():
            _module = module if not copy else deepcopy(module)
            branch.append(_module, link=link)

        return self

    def prepend_for_each(self, module: nn.Module, copy=False, link=None) -> "BlockContainerDict":
        for branch in self.values():
            _module = module if not copy else deepcopy(module)
            branch.prepend(_module, link=link)

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

    def __rich_repr__(self, title=None):
        if not title:
            title = self._get_name()
        table = Table(title=title, title_justify="left")
        branch_row = []

        for branch_name, branch in self.items():
            table.add_column(branch_name, justify="left", no_wrap=True)
            if len(branch) == 1:
                branch_row.append(rich_utils.module_tree(branch[0]))
            else:
                branch_row.append(rich_utils.module_tree(branch))

        table.add_row(*branch_row)

        return table

    def rich_print(self):
        rprint(self.__rich_repr__())

    def _get_name(self) -> str:
        return super()._get_name() if self._name is None else self._name
