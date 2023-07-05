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
from typing import Callable, List, Tuple, Type, TypeVar, Union

import torch
from torch import nn
from typing_extensions import Self

from merlin.models.torch import schema

ModuleType = TypeVar("ModuleType", bound=nn.Module)
PredicateFn = Callable[[ModuleType], bool]


def find(module: nn.Module, to_search: Union[PredicateFn, Type[ModuleType]]) -> List[ModuleType]:
    """
    Traverse a PyTorch Module and find submodules matching a given condition.

    Finding a module-type::
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> find(model, nn.Linear)  # find all Linear layers
        [Linear(in_features=10, out_features=20, bias=True)]

    Finding a module-type with a condition::
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> find(model, lambda x: isinstance(x, nn.Linear) and x.out_features == 10)
        [Linear(in_features=20, out_features=10, bias=True)]

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to traverse.
    to_search : Union[Callable[[ModuleType], bool], Type[ModuleType]]
        The condition to match. Can be either a subclass of nn.Module, in which case
        submodules of that type are searched, or a Callable, which is applied to each
        submodule and should return True for matches.

    Returns
    -------
    List[ModuleType]
        List of matching submodules.

    Raises
    ------
    ValueError
        If `to_search` is neither a subclass of nn.Module nor a Callable.
    """

    if isinstance(to_search, type) and issubclass(to_search, nn.Module):
        predicate = lambda x: isinstance(x, to_search)  # noqa: E731
    elif callable(to_search):
        predicate = to_search
    else:
        raise ValueError("to_search must be either a subclass of nn.Module or a callable.")

    result = []

    def apply_fn(m: nn.Module):
        nonlocal result
        if predicate(m):
            result.append(m)

    module.apply(apply_fn)

    return result


def leaf(module) -> nn.Module:
    """
    Recursively fetch the deepest child module.

    Example usage::
        >>> model = nn.Sequential(nn.Linear(10, 20))
        >>> print(leaf(model))
        Linear(in_features=10, out_features=20, bias=True)

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module to fetch the deepest child from.

    Returns
    -------
    torch.nn.Module
        The deepest child module.

    Raises
    ------
    ValueError
        If any level of the module has more than one child.
    """

    from merlin.models.torch.container import BlockContainer, BlockContainerDict

    containers = (
        nn.Sequential,
        nn.ModuleList,
        nn.ModuleDict,
        BlockContainerDict,
        BlockContainer,
    )

    children = list(module.children())
    if len(children) == 0 or not isinstance(module, containers):
        # If no children, return the module itself (the leaf).
        return module
    elif len(children) == 1:
        child = children[0]

        if hasattr(child, "unwrap"):
            child = child.unwrap()

        if hasattr(child, "leaf"):
            return child.leaf()
        return leaf(child)
    else:
        if isinstance(module, containers) and not hasattr(module, "items"):
            return module[-1]

        # If more than one child, throw an exception.
        raise ValueError(
            f"Module {module} has multiple children, cannot determine the deepest child."
        )


class TraversableMixin:
    def find(self, to_search: Union[PredicateFn, Type[ModuleType]]) -> List[ModuleType]:
        """
        Traverse the current module and find submodules matching a given condition.

        Parameters
        ----------
        to_search : Union[Callable[[ModuleType], bool], Type[ModuleType]]
            The condition to match. Can be either a subclass of nn.Module, in which case
            submodules of that type are searched, or a Callable, which is applied to each
            submodule and should return True for matches.

        Returns
        -------
        List[ModuleType]
            List of matching submodules.
        """
        return find(self, to_search)

    def leaf(self) -> nn.Module:
        """
        Recursively fetch the deepest child module.

        Returns
        -------
        torch.nn.Module
            The deepest child module.
        """
        return leaf(self)

    @torch.jit.ignore
    def select(self, selection: schema.Selection) -> Self:
        return schema.select.dispatched(self, selection)

    @torch.jit.ignore
    def extract(self, selection: schema.Selection) -> Tuple[nn.Module, nn.Module]:
        extraction = schema.select(self, selection)

        return schema.extract.extract(self, selection, extraction), extraction
