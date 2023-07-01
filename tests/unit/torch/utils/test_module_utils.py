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

from typing import Dict, Union

import pytest
import torch
from torch import nn

from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.utils.module_utils import (
    _all_close_dict,
    check_batch_arg,
    is_tabular,
    module_test,
)


class ModuleWithDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]):
        pass


class ModuleWithDictUnion(nn.Module):
    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]):
        pass


class ModuleWithDictUnion2(nn.Module):
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        pass


class ModuleWithBatch(nn.Module):
    def forward(self, x, batch=None):
        pass


class ModuleWithBatchRequired(nn.Module):
    def forward(self, x, batch):
        pass


class ModuleWithoutDict(nn.Module):
    def forward(self, x: torch.Tensor):
        pass


class Test_is_tabular:
    def test_basic(self):
        module_with_dict = ModuleWithDict()
        module_without_dict = ModuleWithoutDict()

        assert is_tabular(module_with_dict)
        assert not is_tabular(module_without_dict)

    def test_union(self):
        module_with_dict_union = ModuleWithDictUnion()
        module_with_dict_union2 = ModuleWithDictUnion2()

        assert is_tabular(module_with_dict_union)
        assert is_tabular(module_with_dict_union2)


class Test_check_batch_arg:
    def test_basic(self):
        module_with_batch = ModuleWithBatch()
        module_without_batch = ModuleWithoutDict()
        module_with_batch_required = ModuleWithBatchRequired()

        assert check_batch_arg(module_with_batch) == (True, False)
        assert check_batch_arg(module_without_batch) == (False, False)
        assert check_batch_arg(module_with_batch_required) == (True, True)


class TestModuleTest:
    def test_simple_module_script(self):
        module = nn.Linear(5, 3)
        input_data = torch.randn(1, 5)

        output = module_test(module, input_data, method="script")

        assert torch.is_tensor(output)

    def test_simple_module_trace(self):
        module = nn.Linear(5, 3)
        input_data = torch.randn(1, 5)

        output = module_test(module, input_data, method="trace")

        assert torch.is_tensor(output)

    def test_unknown_method(self):
        module = nn.Linear(5, 3)
        input_data = torch.randn(1, 5)

        with pytest.raises(ValueError, match="Unknown method: unknown"):
            module_test(module, input_data, method="unknown")

    def test_incompatible_inputs(self):
        module = nn.Linear(5, 3)
        input_data = torch.randn(1, 4)  # Incompatible inputs

        with pytest.raises(RuntimeError, match="Failed to call the module with provided inputs"):
            module_test(module, input_data, method="script")

    def test_cannot_script(self):
        # Define a module that cannot be scripted
        class NotScriptable(nn.Module):
            def forward(self, x):
                return x.tolist()  # tolist() is not scriptable

        module = NotScriptable()
        input_data = torch.randn(1, 5)

        with pytest.raises(RuntimeError, match="Failed to script the module"):
            module_test(module, input_data, method="script", schema_trace=False)

    def test_output_dict(self):
        # Define a module that returns a dictionary
        class DictOutput(nn.Module):
            def forward(self, x):
                return {"output": x}

        module = DictOutput()
        input_data = torch.randn(1, 5)

        output = module_test(module, input_data, method="script")

        assert isinstance(output, dict)
        assert "output" in output

    def test_output_tuple(self):
        # Define a module that returns a tuple
        class TupleOutput(nn.Module):
            def forward(self, x):
                return x, x

        module = TupleOutput()
        input_data = torch.randn(1, 5)

        output = module_test(module, input_data, method="script")

        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_module_test_output_batch(self):
        # Define a module that returns a Batch instance
        class BatchOutput(nn.Module):
            def forward(self, x):
                features = {"output": x}
                targets = {"target": x}
                sequences = Sequence(
                    lengths={"length": torch.tensor([5])}, masks={"mask": torch.ones(1, 5)}
                )
                return Batch(features=features, targets=targets, sequences=sequences)

        module = BatchOutput()
        input_data = torch.randn(1, 5)

        output = module_test(module, input_data, method="script")

        assert isinstance(output, Batch)
        assert "output" in output.features
        assert "target" in output.targets
        assert "length" in output.sequences.lengths
        assert "mask" in output.sequences.masks

    def test_all_close_dict_mismatch(self):
        left = {"key": torch.tensor(1.0)}
        right = {"key": torch.tensor(2.0)}  # Mismatched value

        with pytest.raises(
            ValueError, match="The outputs of the original and scripted modules are not the same"
        ):
            _all_close_dict(left, right)

    def test_module_test_output_tensor_mismatch(self):
        # Define a module that returns different tensor each time
        class TensorOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, x):
                self.counter += 1
                return x + self.counter

        module = TensorOutput()
        input_data = torch.tensor([1.0])

        with pytest.raises(
            ValueError, match="The outputs of the original and scripted modules are not the same"
        ):
            module_test(module, input_data, method="script", schema_trace=False)

    def test_module_test_output_tuple_mismatch(self):
        # Define a module that returns a tuple with different tensors each time
        class TupleOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, x):
                self.counter += 1
                return (x, x + self.counter)

        module = TupleOutput()
        input_data = torch.tensor([1.0])

        with pytest.raises(
            ValueError, match="The outputs of the original and scripted modules are not the same"
        ):
            module_test(module, input_data, method="script")
