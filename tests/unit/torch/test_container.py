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

import pytest
import torch.nn as nn

from merlin.models.torch.container import BlockContainer
from merlin.models.torch.utils import torchscript_utils


class TestBlockContainer:
    def setup_method(self):
        self.block_container = BlockContainer(name="test_container")

    def test_init(self):
        assert isinstance(self.block_container, BlockContainer)
        assert self.block_container._name == "test_container"

    def test_append(self):
        module = nn.Linear(20, 30)
        self.block_container.append(module)
        assert len(self.block_container) == 1

    def test_prepend(self):
        module1 = nn.Linear(20, 30)
        module2 = nn.Linear(30, 40)
        self.block_container.append(module1)
        self.block_container.prepend(module2)
        assert len(self.block_container) == 2
        assert isinstance(self.block_container[0], nn.Linear)

    def test_insert(self):
        module1 = nn.Linear(20, 30)
        module2 = nn.Linear(30, 40)
        self.block_container.append(module1)
        self.block_container.insert(0, module2)
        assert len(self.block_container) == 2
        assert isinstance(self.block_container[0], nn.Linear)

    def test_len(self):
        module = nn.Linear(20, 30)
        self.block_container.append(module)
        assert len(self.block_container) == 1

    def test_getitem(self):
        module = nn.Linear(20, 30)
        self.block_container.append(module)
        assert isinstance(self.block_container[0], nn.Linear)

    def test_setitem(self):
        module1 = nn.Linear(20, 30)
        module2 = nn.Linear(30, 40)
        self.block_container.append(module1)
        self.block_container[0] = module2
        assert isinstance(self.block_container[0], nn.Linear)

    def test_delitem(self):
        module = nn.Linear(20, 30)
        self.block_container.append(module)
        del self.block_container[0]
        assert len(self.block_container) == 0

    def test_add(self):
        other_block_container = BlockContainer()
        module = nn.Linear(20, 30)
        other_block_container.append(module)
        self.block_container += other_block_container
        assert len(self.block_container) == 1

    def test_repr(self):
        assert repr(self.block_container)

    def test_none_input(self):
        with pytest.raises(ValueError):
            self.block_container.append(None)

    def test_non_module_input(self):
        with pytest.raises(ValueError):
            self.block_container.append("Not a Module")

    def test_getitem_out_of_range(self):
        with pytest.raises(IndexError):
            _ = self.block_container[0]

    def test_delitem_out_of_range(self):
        with pytest.raises(IndexError):
            del self.block_container[0]

    def test_setitem_out_of_range(self):
        module = nn.Linear(20, 30)
        with pytest.raises(IndexError):
            self.block_container[0] = module

    def test_unwrap(self):
        module = nn.Linear(20, 30)
        self.block_container.append(module)
        unwrapped = self.block_container.unwrap()
        assert isinstance(unwrapped, nn.ModuleList)

    def test_wrap_module_with_module(self):
        module = nn.Linear(20, 30)
        wrapped = self.block_container.wrap_module(module)
        assert isinstance(wrapped, torchscript_utils.TorchScriptWrapper)

    def test_wrap_module_with_blockcontainer(self):
        other_block_container = BlockContainer()
        module = nn.Linear(20, 30)
        other_block_container.append(module)
        wrapped = self.block_container.wrap_module(other_block_container)
        assert isinstance(wrapped, BlockContainer)

    def test_add_module(self):
        module = nn.Linear(20, 30)
        self.block_container.add_module("test", module)
        assert "test" in self.block_container._modules

    def test_get_name(self):
        assert self.block_container._get_name() == "test_container"
