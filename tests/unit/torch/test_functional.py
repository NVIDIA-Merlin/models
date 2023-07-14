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

from typing import Iterable, Tuple

import pytest
import torch.nn as nn

import merlin.models.torch as mm
from merlin.models.torch.container import BlockContainer
from merlin.models.torch.functional import _create_list_wrapper


class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        return x + self.linear2(self.linear1(x))


def add_relu(x):
    if isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_named(x, name=None, to_replace="linear1"):
    if name == to_replace and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_first(x, i=None):
    if i == 0 and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


class TestMapModule:
    def test_map_identity(self):
        # Test mapping an identity function
        module = nn.Linear(10, 10)
        identity = lambda x: x  # noqa: E731
        assert mm.map(module, identity) is module

    def test_map_transform(self):
        # Test mapping a transform function
        module = nn.Linear(10, 10)
        transformed_module = mm.map(module, add_relu)
        assert isinstance(transformed_module[0], nn.Linear)
        assert isinstance(transformed_module[1], nn.ReLU)

    def test_walk_custom_module(self):
        mlp = CustomMLP()
        with_relu = mm.walk(mlp, add_relu)
        assert isinstance(with_relu.linear1, nn.Sequential)
        assert isinstance(with_relu.linear2, nn.Sequential)

        for fn in [add_relu_named, add_relu_first]:
            with_relu_first = mm.walk(mlp, fn)
            assert isinstance(with_relu_first.linear1, nn.Sequential)
            assert isinstance(with_relu_first.linear2, nn.Linear)


class TestMapModuleList:
    def test_map_identity(self):
        # Test mapping an identity function
        modules = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        identity = lambda x: x  # noqa: E731
        mapped = mm.map(modules, identity)
        assert all(m1 == m2 for m1, m2 in zip(modules, mapped))

    @pytest.mark.parametrize("wrapper", [nn.Sequential, nn.ModuleList])
    def test_map_with_index(self, wrapper):
        # Test mapping a function that uses the index
        modules = _create_list_wrapper(wrapper(), [nn.Linear(10, 10) for _ in range(5)])

        def add_index(x, i):
            return nn.Linear(10 + i, 10 + i)

        new_modules = mm.map(modules, add_index)
        assert isinstance(new_modules, wrapper)
        for i, module in enumerate(new_modules):
            assert isinstance(module, nn.Linear)
            assert module.in_features == 10 + i
            assert module.out_features == 10 + i


class TestMapModuleDict:
    def test_map_module_dict(self):
        # Define a simple transformation function
        def transformation(module: nn.Module, name: str = "", **kwargs) -> nn.Module:
            if isinstance(module, nn.Linear):
                return nn.Linear(20, 10)
            return module

        # Define a ModuleDict of modules
        module_dict = nn.ModuleDict({"linear1": nn.Linear(10, 10), "linear2": nn.Linear(10, 10)})

        # Apply map_module_dict
        new_module_dict = mm.map(module_dict, transformation)

        # Assert that the transformation has been applied correctly
        for module in new_module_dict.values():
            assert isinstance(module, nn.Linear)
            assert module.in_features == 20
            assert module.out_features == 10


class TestContainerMixin:
    @pytest.fixture
    def container(self) -> BlockContainer:
        return BlockContainer(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))

    def test_filter(self, container):
        filtered = container.filter(lambda m: isinstance(m, nn.ReLU))
        assert isinstance(filtered, BlockContainer)
        assert len(filtered) == 1
        assert isinstance(filtered[0], nn.ReLU)

    def test_filter_recurse(self, container):
        def func(module):
            return isinstance(module, nn.Linear)

        filtered = BlockContainer(container).filter(func, recurse=True)
        assert isinstance(filtered, BlockContainer)
        assert len(filtered) == 1
        assert len(filtered[0]) == 2
        assert isinstance(filtered[0][0], nn.Linear)
        assert isinstance(filtered[0][1], nn.Linear)

    def test_flatmap(self, container):
        def func(module):
            return BlockContainer(*([module] * 2))

        flat_mapped = container.flatmap(func)
        assert isinstance(flat_mapped, BlockContainer)
        assert len(flat_mapped) == 6

    def test_flatmap_non_callable(self, container):
        with pytest.raises(TypeError):
            container.flatmap(123)

    def test_forall(self, container):
        def func(module):
            return isinstance(module, nn.Module)

        assert container.forall(func)

    def test_forall_recurse(self, container):
        def func(module):
            return isinstance(module, nn.ReLU)

        assert not BlockContainer(container).forall(func, recurse=True)
        assert BlockContainer(container).forall(lambda x: True, recurse=True)

    def test_map(self, container):
        def func(module):
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)
            return module

        mapped = container.map(func)
        assert isinstance(mapped, BlockContainer)
        assert len(mapped) == 3
        assert isinstance(mapped[0], nn.Conv2d)
        assert isinstance(mapped[1], nn.ReLU)
        assert isinstance(mapped[2], nn.Conv2d)

    def test_map_recurse(self, container):
        def func(module):
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)
            return module

        mapped = BlockContainer(container).map(func, recurse=True)
        assert isinstance(mapped, BlockContainer)
        assert len(mapped) == 1
        assert len(mapped[0]) == 3
        assert isinstance(mapped[0][0], nn.Conv2d)
        assert isinstance(mapped[0][1], nn.ReLU)
        assert isinstance(mapped[0][2], nn.Conv2d)

    def test_mapi(self, container):
        def func(module, idx):
            assert idx in [0, 1, 2]

            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)
            return module

        mapped = container.mapi(func)
        assert isinstance(mapped, BlockContainer)
        assert len(mapped) == 3
        assert isinstance(mapped[0], nn.Conv2d)
        assert isinstance(mapped[1], nn.ReLU)
        assert isinstance(mapped[2], nn.Conv2d)

    def test_mapi_recurse(self, container):
        def func(module, idx):
            assert idx in [0, 1, 2]
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)
            return module

        mapped = BlockContainer(container).mapi(func, recurse=True)
        assert isinstance(mapped, BlockContainer)
        assert len(mapped) == 1
        assert len(mapped[0]) == 3
        assert isinstance(mapped[0][0], nn.Conv2d)
        assert isinstance(mapped[0][1], nn.ReLU)
        assert isinstance(mapped[0][2], nn.Conv2d)

    def test_choose(self, container):
        def func(module):
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)

        chosen = container.choose(func)
        assert isinstance(chosen, BlockContainer)
        assert len(chosen) == 2
        assert isinstance(chosen[0], nn.Conv2d)

    def test_choose_recurse(self, container):
        def func(module):
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)

        chosen = BlockContainer(container).choose(func, recurse=True)
        assert isinstance(chosen, BlockContainer)
        assert len(chosen) == 1
        assert len(chosen[0]) == 2
        assert isinstance(chosen[0][0], nn.Conv2d)
        assert isinstance(chosen[0][1], nn.Conv2d)

    def test_walk(self, container: BlockContainer):
        def func(module):
            if isinstance(module, nn.Linear):
                return nn.Conv2d(3, 3, 3)
            return module

        walked = BlockContainer(container).walk(func)
        assert isinstance(walked, BlockContainer)
        assert len(walked) == 1
        assert len(walked[0]) == 3
        assert isinstance(walked[0][0], nn.Conv2d)
        assert isinstance(walked[0][1], nn.ReLU)
        assert isinstance(walked[0][2], nn.Conv2d)

    def test_zip(self, container: BlockContainer):
        other = BlockContainer(nn.Conv2d(3, 3, 3), nn.ReLU(), nn.Linear(5, 2))
        zipped = lambda: container.zip(other)  # noqa: E731
        assert isinstance(zipped(), Iterable)
        assert len(list(zipped())) == 3
        assert isinstance(list(zipped())[0], Tuple)
        assert isinstance(list(zipped())[0][0], nn.Linear)
        assert isinstance(list(zipped())[0][1], nn.Conv2d)

    def test_add(self, container):
        new_module = nn.Linear(5, 2)
        new_container = container + new_module
        assert isinstance(new_container, BlockContainer)
        assert len(new_container) == 4
        assert isinstance(new_container[3], nn.Linear)

        _container = container + container
        assert len(_container) == 6

    def test_radd(self, container):
        new_module = nn.Linear(5, 2)
        new_container = new_module + container
        assert isinstance(new_container, BlockContainer)
        assert len(new_container) == 4
        assert isinstance(new_container[0], nn.Linear)

    def test_freeze(self, container):
        container.freeze()
        for param in container.parameters():
            assert not param.requires_grad

    def test_unfreeze(self, container):
        container.unfreeze()
        for param in container.parameters():
            assert param.requires_grad
