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
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.container import BlockContainer


class Block(BlockContainer):
    """A base-class that calls it's modules sequentially.

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.
    """

    def __init__(self, *module: nn.Module, name: Optional[str] = None):
        super().__init__(*module, name=name)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        """
        Forward pass through the block. Applies each contained module sequentially on the input.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input data as a tensor or a dictionary of tensors.
        batch : Optional[Batch], default = None
            Optional batch of data. If provided, it is used by the `module`s.

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            The output of the block after processing the input.
        """
        for module in self.values:
            inputs = module(inputs, batch=batch)

        return inputs

    def repeat(self, n: int = 1, name=None) -> "Block":
        """
        Creates a new block by repeating the current block `n` times.
        Each repetition is a deep copy of the current block.

        Parameters
        ----------
        n : int
            The number of times to repeat the current block.
        name : Optional[str], default = None
            The name for the new block. If None, no name is assigned.

        Returns
        -------
        Block
            The new block created by repeating the current block `n` times.
        """
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n < 1:
            raise ValueError("n must be greater than 0")

        repeats = [self.copy() for _ in range(n - 1)]

        return Block(self, *repeats, name=name)

    def copy(self) -> "Block":
        """
        Creates a deep copy of the current block.

        Returns
        -------
        Block
            The copy of the current block.
        """
        return deepcopy(self)