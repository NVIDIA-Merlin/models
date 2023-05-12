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

from typing import Dict

import pytest
import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.utils.torchscript_utils import TorchScriptWrapper


class SimpleModule(nn.Module):
    def forward(self, x: torch.Tensor):
        return x


class SimpleModuleBatch(nn.Module):
    def forward(self, x: torch.Tensor, batch: Batch = None):
        return x


class SimpleModuleRequiresBatch(nn.Module):
    def forward(self, x: torch.Tensor, batch: Batch):
        return x


class SimpleModuleDictInput(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor], batch: Batch = None):
        return x


class TestTorchScriptWrapper:
    def test_init(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        assert wrapper.to_wrap == simple_module
        assert wrapper.unwrap() == simple_module
        assert not wrapper.accepts_batch
        assert not wrapper.requires_batch
        assert not wrapper.accepts_dict

    def test_forward_no_batch_no_dict(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        inputs = torch.randn(1, 5)
        output = wrapper.forward(inputs)

        assert torch.equal(output, inputs)

    def test_forward_accepts_batch_not_dict(self):
        simple_module_batch = SimpleModuleBatch()
        wrapper = TorchScriptWrapper(simple_module_batch)

        inputs = torch.randn(1, 5)
        batch = Batch({})
        output = wrapper.forward(inputs, batch)

        assert torch.equal(output, inputs)

    def test_forward_requires_batch_not_dict(self):
        simple_module_requires_batch = SimpleModuleRequiresBatch()
        wrapper = TorchScriptWrapper(simple_module_requires_batch)

        inputs = torch.randn(1, 5)
        batch = Batch({})
        output = wrapper.forward(inputs, batch)

        assert torch.equal(output, inputs)

    def test_forward_no_batch_accepts_dict(self):
        simple_module_dict_input = SimpleModuleDictInput()
        wrapper = TorchScriptWrapper(simple_module_dict_input)

        inputs = {"x": torch.randn(1, 5)}
        output = wrapper.forward(inputs)

        assert torch.equal(output["x"], inputs["x"])

    def test_unwrap(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        assert wrapper.unwrap() == simple_module

    def test_forward_requires_batch_no_batch_provided(self):
        simple_module_requires_batch = SimpleModuleRequiresBatch()
        wrapper = TorchScriptWrapper(simple_module_requires_batch)

        inputs = torch.randn(1, 5)

        with pytest.raises(RuntimeError, match="batch is required for this module"):
            wrapper.forward(inputs)

    def test_forward_accepts_batch_dict_input_but_tensor_provided(self):
        simple_module_dict_input = SimpleModuleDictInput()
        wrapper = TorchScriptWrapper(simple_module_dict_input)
        wrapper.accepts_batch = True  # Force accepts_batch to True

        inputs = torch.randn(1, 5)
        batch = Batch({})

        with pytest.raises(RuntimeError, match="Expected a dictionary, but got a tensor instead."):
            wrapper.forward(inputs, batch)

    def test_forward_accepts_batch_tensor_input_but_dict_provided(self):
        simple_module_batch = SimpleModuleBatch()
        wrapper = TorchScriptWrapper(simple_module_batch)
        wrapper.accepts_dict = False  # Force accepts_dict to False

        inputs = {"x": torch.randn(1, 5)}
        batch = Batch({})

        with pytest.raises(RuntimeError, match="Expected a tensor, but got a dictionary instead."):
            wrapper.forward(inputs, batch)

    def test_forward_accepts_tensor_but_dict_provided(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        inputs = {"x": torch.randn(1, 5)}

        with pytest.raises(RuntimeError, match="Expected a tensor, but got a dictionary instead."):
            wrapper.forward(inputs)

    def test_forward_accepts_dict_but_tensor_provided(self):
        simple_module_dict_input = SimpleModuleDictInput()
        wrapper = TorchScriptWrapper(simple_module_dict_input)

        inputs = torch.randn(1, 5)

        with pytest.raises(RuntimeError, match="Expected a dictionary, but got a tensor instead."):
            wrapper.forward(inputs)

    def test_forward_accepts_tensor_and_batch_but_dict_provided(self):
        simple_module_accepts_batch = SimpleModuleBatch()
        wrapper = TorchScriptWrapper(simple_module_accepts_batch)

        inputs = {"x": torch.randn(1, 5)}
        batch = Batch({})  # Assuming a Batch can be initialized this way

        with pytest.raises(RuntimeError, match="Expected a tensor, but got a dictionary instead."):
            wrapper.forward(inputs, batch)

    def test_forward_accepts_dict_and_batch_but_tensor_provided(self):
        simple_module_accepts_dict_and_batch = SimpleModuleDictInput()
        wrapper = TorchScriptWrapper(simple_module_accepts_dict_and_batch)

        inputs = torch.randn(1, 5)
        batch = Batch({})  # Assuming a Batch can be initialized this way

        with pytest.raises(RuntimeError, match="Expected a dictionary, but got a tensor instead."):
            wrapper.forward(inputs, batch)

    def test_getattr_from_wrapped_module(self):
        simple_module = SimpleModule()
        simple_module.new_attribute = "test"
        wrapper = TorchScriptWrapper(simple_module)

        assert wrapper.new_attribute == "test"

    def test_init_non_module(self):
        with pytest.raises(
            ValueError, match="Expected a nn.Module, but got something else instead."
        ):
            TorchScriptWrapper("not a module")

    def test_repr(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        assert repr(wrapper) == repr(simple_module)

    def test_getattr_non_existent(self):
        simple_module = SimpleModule()
        wrapper = TorchScriptWrapper(simple_module)

        with pytest.raises(AttributeError):
            wrapper.non_existent_attribute
