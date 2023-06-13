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

import textwrap
from copy import deepcopy
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.container import BlockContainer, BlockContainerDict
from merlin.models.torch.link import Link, LinkType
from merlin.models.torch.registry import registry
from merlin.models.torch.utils.schema_utils import input_schema, output_schema
from merlin.models.utils.registry import RegistryMixin
from merlin.schema import Schema


class Block(BlockContainer, RegistryMixin):
    """A base-class that calls it's modules sequentially.

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

    def __init__(self, *inputs: Union[nn.Module, Dict[str, nn.Module]]):
        pre = BlockContainer()
        branches = BlockContainerDict(*inputs)
        post = BlockContainer()

        super().__init__()

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

            # We need to check whether or not the branch container has a forward-method
            # This is a bit of a hack, but this seems to work in torchscript as well.
            if hasattr(branch_container, "registry"):
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

    def replace(self, pre=None, branches=None, post=None) -> "ParallelBlock":
        """Replaces the pre-processing, branching and post-processing stages.

        Parameters
        ----------
        pre : Optional[BlockContainer], default=None
            The pre-processing stage.
        branches : Optional[BlockContainerDict], default=None
            The branching stage.
        post : Optional[BlockContainer], default=None
            The post-processing stage.

        Returns
        -------
        ParallelBlock
            The current object itself.
        """
        output = ParallelBlock(branches if branches is not None else self.branches)
        output.pre = pre if pre is not None else self.pre
        output.post = post if post is not None else self.post

        return output

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

    def __bool__(self) -> bool:
        return bool(self.branches) or bool(self.pre) or bool(self.post)

    def __eq__(self, other) -> bool:
        return self.pre == other.pre and self.branches == other.branches and self.post == other.post

    def __hash__(self) -> int:
        return hash((self.pre, self.branches, self.post))

    def __repr__(self) -> str:
        indent_str = "    "
        branches = repr(self.branches)[len("BlockContainerDict") :]

        pre = ""
        if self.pre:
            pre = textwrap.indent("(pre): " + repr(self.pre), indent_str)

        post = ""
        if self.post:
            post = textwrap.indent("(post): " + repr(self.post), indent_str)

        if self.pre or self.post:
            branches = textwrap.indent("(branches): " + branches, indent_str)
            output = ""
            for o in [pre, branches, post]:
                if o:
                    output += "\n" + o

            return f"{self._get_name()}({output}\n)"

        return self._get_name() + branches


def get_pre(module: nn.Module) -> BlockContainer:
    if hasattr(module, "pre"):
        return module.pre

    if isinstance(module, BlockContainer):
        return get_pre(module[0])

    return BlockContainer()


def set_pre(module: nn.Module, pre: BlockContainer):
    if hasattr(module, "pre"):
        module.pre = pre

    if isinstance(module, BlockContainer):
        return set_pre(module[0], pre)


@input_schema.register(BlockContainer)
def _(module: BlockContainer, input: Schema):
    return input_schema(module[0], input) if module else input


@input_schema.register(ParallelBlock)
def _(module: ParallelBlock, input: Schema):
    if module.pre:
        return input_schema(module.pre)

    schema = Schema()
    for branch in module.branches.values():
        schema += input_schema(branch, input)

    return schema


@output_schema.register(ParallelBlock)
def _(module: ParallelBlock, input: Schema):
    if module.post:
        return output_schema(module.post, input)

    output = Schema()
    for name, branch in module.branches.items():
        branch_schema = output_schema(branch, input)

        if len(branch_schema) == 1 and branch_schema.first.name == "output":
            branch_schema = Schema([branch_schema.first.with_name(name)])

        output += branch_schema

    return output


@output_schema.register(BlockContainer)
def _(module: BlockContainer, input: Schema):
    return output_schema(module[-1], input) if module else input
