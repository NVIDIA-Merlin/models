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
from merlin.models.torch.container import BlockContainer, BlockContainerDict
from merlin.models.torch.link import Link, LinkType
from merlin.models.torch.registry import registry
from merlin.models.torch.utils.schema_utils import SchemaTrackingMixin
from merlin.models.utils.registry import RegistryMixin


class Block(BlockContainer, SchemaTrackingMixin, RegistryMixin):
    """A base-class that calls its modules sequentially.

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.
    track_schema : bool, default = True
        If True, the schema of the output tensors are tracked.
    """

    registry = registry

    def __init__(self, *module: nn.Module, name: Optional[str] = None, track_schema: bool = True):
        super().__init__(*module, name=name)
        if track_schema:
            self._register_schema_tracking_hook()

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

    def repeat(self, n: int = 1, link: Optional[LinkType] = None, name=None) -> "Block":
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
        if link:
            parsed_link = Link.parse(link)
            repeats = [parsed_link.copy().setup_link(repeat) for repeat in repeats]

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


class ParallelBlock(Block):
    """A base-class that calls its modules in parallel.

    A ParallelBlock contains multiple branches that will be executed
    in parallel. Each branch can contain multiple modules, and
    the outputs of all branches are collected into a dictionary.

    If a branch returns a dictionary of tensors instead of a single tensor,
    it will be flattened into the output dictionary. This ensures the output-type
    is always Dict[str, torch.Tensor].

    Example usage::
        >>> parallel_block = ParallelBlock({"a": nn.LazyLinear(2), "b": nn.LazyLinear(2)})
        >>> x = torch.randn(2, 2)
        >>> output = parallel_block(x)
        >>> # The output is a dictionary containing the output of each branch
        >>> print(output)
        {
            'a': tensor([[-0.0801,  0.0436],
                        [ 0.1235, -0.0318]]),
            'b': tensor([[ 0.0918, -0.0274],
                        [-0.0652,  0.0381]])
        }

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.
    track_schema : bool, default = True
        If True, the schema of the output tensors are tracked.
    """

    def __init__(self, *inputs: Union[nn.Module, Dict[str, nn.Module]], track_schema: bool = True):
        pre = BlockContainer(name="pre")
        branches = BlockContainerDict(*inputs)
        post = BlockContainer(name="post")

        super().__init__(track_schema=track_schema)

        self.pre = pre
        self.branches = branches
        self.post = post

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        """Forward pass through the block.

        The steps are as follows:
        1. Pre-processing stage: Applies each module in the pre-processing stage sequentially.
        2. Branching stage: Applies each module in each branch sequentially.
        3. Post-processing stage: Applies each module in the post-processing stage sequentially.

        If a branch returns a dictionary of tensors instead of a single tensor,
        it will be flattened into the output dictionary. This ensures the output-type
        is always Dict[str, torch.Tensor].

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input tensor or dictionary of tensors.
        batch : Optional[Batch], default=None
            An optional batch of data.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors.
        """
        for module in self.pre.values:
            inputs = module(inputs, batch=batch)

        outputs = {}
        for name, branch_container in self.branches.items():
            branch = inputs

            if hasattr(branch_container, "branches"):
                branch = branch_container(branch, batch=batch)
            else:
                for module in branch_container.values:
                    branch = module(branch, batch=batch)

            if isinstance(branch, torch.Tensor):
                branch_dict = {name: branch}
            elif torch.jit.isinstance(branch, Dict[str, torch.Tensor]):
                branch_dict = branch
            else:
                raise TypeError(
                    f"Branch output must be a tensor or a dictionary of tensors. Got {type(branch)}"
                )

            for key in branch_dict.keys():
                if key in outputs:
                    raise RuntimeError(f"Duplicate output name: {key}")

            outputs.update(branch_dict)

        for module in self.post.values:
            outputs = module(outputs, batch=batch)

        return outputs

    def append(self, module: nn.Module, link: Optional[LinkType] = None):
        """Appends a module to the post-processing stage.

        Parameters
        ----------
        module : nn.Module
            The module to append.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """

        self.post.append(module, link=link)

        return self

    def prepend(self, module: nn.Module):
        self.pre.prepend(module)

        return self

    def append_to(self, name: str, module: nn.Module, link: Optional[LinkType] = None):
        """Appends a module to a specified branch.

        Parameters
        ----------
        name : str
            The name of the branch.
        module : nn.Module
            The module to append.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """

        self.branches[name].append(module, link=link)

        return self

    def prepend_to(self, name: str, module: nn.Module, link: Optional[LinkType] = None):
        """Prepends a module to a specified branch.

        Parameters
        ----------
        name : str
            The name of the branch.
        module : nn.Module
            The module to prepend.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """
        self.branches[name].prepend(module, link=link)

        return self

    def append_for_each(self, module: nn.Module, shared=False, link: Optional[LinkType] = None):
        """Appends a module to each branch.

        Parameters
        ----------
        module : nn.Module
            The module to append.
        shared : bool, default=False
            If True, the same module is shared across all branches.
            Otherwise a deep copy of the module is used in each branch.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """

        self.branches.append_for_each(module, shared=shared, link=link)

        return self

    def prepend_for_each(self, module: nn.Module, shared=False, link: Optional[LinkType] = None):
        """Prepends a module to each branch.

        Parameters
        ----------
        module : nn.Module
            The module to prepend.
        shared : bool, default=False
            If True, the same module is shared across all branches.
            Otherwise a deep copy of the module is used in each branch.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """

        self.branches.prepend_for_each(module, shared=shared, link=link)

        return self

    def __getitem__(self, idx: Union[slice, int, str]):
        if isinstance(idx, str) and idx in self.branches:
            return self.branches[idx]

        if idx == 0:
            return self.pre

        if idx == -1 or idx == 2:
            return self.post

        raise IndexError(f"Index {idx} is out of range for {self.__class__.__name__}")

    def __len__(self):
        return len(self.branches)

    def __contains__(self, name):
        return name in self.branches
