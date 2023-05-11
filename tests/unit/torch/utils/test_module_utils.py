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
