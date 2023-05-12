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

import torch
from torch import nn

from merlin.models.torch.utils.module_utils import check_batch_arg, is_tabular


class ModuleWithDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]):
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


class Test_check_batch_arg:
    def test_basic(self):
        module_with_batch = ModuleWithBatch()
        module_without_batch = ModuleWithoutDict()
        module_with_batch_required = ModuleWithBatchRequired()

        assert check_batch_arg(module_with_batch) == (True, False)
        assert check_batch_arg(module_without_batch) == (False, False)
        assert check_batch_arg(module_with_batch_required) == (True, True)
