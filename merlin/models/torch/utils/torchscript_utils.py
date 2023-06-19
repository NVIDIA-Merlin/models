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

from typing import Dict, Final, Optional, Set, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.utils import module_utils


class TorchScriptWrapper(nn.Module):
    """
    A wrapper class for PyTorch `nn.Module` to make it compatible with TorchScript.

    It checks and determines the appropriate input format
    (tensor or dictionary) and batch requirements for the wrapped module,
    ensuring that these are fulfilled and appropriately passed for TorchScript's
    static typing system.

    Parameters
    ----------
    to_wrap : nn.Module
        The PyTorch module to be wrapped.

    Attributes
    ----------
    accepts_batch : bool
        A boolean indicating whether the wrapped module accepts batch as an argument.
    requires_batch : bool
        A boolean indicating whether the wrapped module requires batch as an argument.
    accepts_dict : bool
        A boolean indicating whether the wrapped module can take a dictionary as input.
    to_wrap : nn.Module
        The PyTorch module to be wrapped.

    """

    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    accepts_dict: Final[bool]

    def __init__(self, to_wrap: nn.Module):
        super().__init__()

        if not isinstance(to_wrap, nn.Module):
            raise ValueError("Expected a nn.Module, but got something else instead.")

        self.to_wrap = to_wrap
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)
        self.accepts_dict = module_utils.is_tabular(to_wrap)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        """Forward pass through the wrapped module with appropriate torchscript type-checking.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input tensor(s) for the model.
        batch : Optional[Batch]
            An optional batch object.

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            The output of the model.
        """

        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    # else-clause is needed to make torchscript happy
                    _batch = batch if batch is not None else Batch({})

                    if not self.accepts_dict:
                        if not torch.jit.isinstance(inputs, torch.Tensor):
                            raise RuntimeError("Expected a tensor, but got a dictionary instead.")
                        x: torch.Tensor = (
                            inputs if isinstance(inputs, torch.Tensor) else inputs["x"]
                        )

                        return self.to_wrap(x, batch=_batch)

                    else:
                        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                            raise RuntimeError("Expected a dictionary, but got a tensor instead.")
                        x: Dict[str, torch.Tensor] = inputs

                        return self.to_wrap(x, batch=_batch)

        if not self.accepts_dict:
            if not torch.jit.isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["x"]

            return self.to_wrap(x)

        else:
            if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                raise RuntimeError("Expected a dictionary, but got a tensor instead.")
            x: Dict[str, torch.Tensor] = inputs

            return self.to_wrap(x)

    def unwrap(self) -> nn.Module:
        """
        Returns
        -------
        nn.Module
            The original PyTorch module.
        """
        return self.to_wrap

    def __getattr__(self, name: str):
        if name in [
            "accepts_dict",
            "accepts_batch",
            "requires_batch",
            "output_schema",
            "to_wrap",
            "unwrap",
        ]:
            return super().__getattr__(name)

        # Otherwise, return the attribute from the to_wrap module
        return getattr(self.to_wrap, name)

    def __repr__(self):
        return self.to_wrap.__repr__()

    def __eq__(self, value) -> bool:
        if not isinstance(value, TorchScriptWrapper):
            return self.to_wrap == value

        return self.to_wrap == value.to_wrap

    def __hash__(self):
        return hash(self.to_wrap)

    def named_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        return self.to_wrap.named_modules(memo, prefix, remove_duplicate)

    def _apply(self, fn):
        return self.to_wrap._apply(fn)
