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

import inspect
import textwrap
from copy import deepcopy
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import torch
from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.container import BlockContainer, BlockContainerDict
from merlin.models.torch.registry import registry
from merlin.models.torch.utils.traversal_utils import TraversableMixin
from merlin.models.utils.registry import RegistryMixin
from merlin.schema import Schema

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


@runtime_checkable
class HasKeys(Protocol):
    def keys(self):
        ...


class Block(BlockContainer, RegistryMixin, TraversableMixin):
    """A base-class that calls it's modules sequentially.

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.
    """

    registry = registry

    def __init__(self, *module: nn.Module, name: Optional[str] = None):
        super().__init__(*module, name=name)

    def forward(self, inputs: TensorOrDict, batch: Optional[Batch] = None):
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
        return repeat(self, n, name=name)

    def repeat_parallel(self, n: int = 1, name=None) -> "ParallelBlock":
        return repeat_parallel(self, n, name=name)

    def repeat_parallel_like(self, like: HasKeys, agg=None) -> "ParallelBlock":
        return repeat_parallel_like(self, like, agg=agg)

    def copy(self) -> "Block":
        """
        Creates a deep copy of the current block.

        Returns
        -------
        Block
            The copy of the current block.
        """
        return deepcopy(self)

    @torch.jit.ignore
    def select(self, selection: schema.Selection) -> "Block":
        return _select_block(self, selection)

    @torch.jit.ignore
    def extract(self, selection: schema.Selection) -> Tuple[nn.Module, nn.Module]:
        selected = self.select(selection)
        return _extract_block(self, selection, selected), selected


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
        pre = Block()
        branches = BlockContainerDict(*inputs, block_cls=Block)
        post = Block()

        super().__init__()

        self.pre = pre
        self.branches = branches
        self.post = post

    def forward(self, inputs: TensorOrDict, batch: Optional[Batch] = None):
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
        _inputs = self.pre(inputs, batch=batch)

        outputs = {}
        for name, branch_container in self.branches.items():
            branch_out = branch_container(_inputs, batch=batch)

            if torch.jit.isinstance(branch_out, torch.Tensor):
                if name in outputs:
                    raise RuntimeError(f"Duplicate output name: {name}")

                outputs.update({name: branch_out})
            elif torch.jit.isinstance(branch_out, Dict[str, torch.Tensor]):
                for key in branch_out.keys():
                    if key in outputs:
                        raise RuntimeError(f"Duplicate output name: {key}")

                outputs.update(branch_out)
            elif torch.jit.isinstance(branch_out, Batch):
                _flattened_batch: Dict[str, torch.Tensor] = branch_out.flatten_as_dict(batch)
                outputs.update(_flattened_batch)
            else:
                raise TypeError(
                    f"Branch output must be a tensor or a dictionary of tensors. Got {_inputs}"
                )

        outputs = self.post(outputs, batch=batch)

        return outputs

    def append(self, module: nn.Module):
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

        self.post.append(module)

        return self

    def prepend(self, module: nn.Module):
        self.pre.prepend(module)

        return self

    def append_to(self, name: str, module: nn.Module):
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

        self.branches[name].append(module)

        return self

    def prepend_to(self, name: str, module: nn.Module):
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
        self.branches[name].prepend(module)

        return self

    def append_for_each(self, module: nn.Module, shared=False):
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

        self.branches.append_for_each(module, shared=shared)

        return self

    def prepend_for_each(self, module: nn.Module, shared=False):
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

        self.branches.prepend_for_each(module, shared=shared)

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

    def keys(self):
        return self.branches.keys()

    def leaf(self) -> nn.Module:
        if self.pre:
            raise ValueError("Cannot call leaf() on a ParallelBlock with a pre-processing stage")

        if len(self.branches) != 1:
            raise ValueError("Cannot call leaf() on a ParallelBlock with multiple branches")

        first = list(self.branches.values())[0]
        return first.leaf()

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
        if not isinstance(other, ParallelBlock):
            return False

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


class ResidualBlock(Block):
    """
    A block that applies each contained module sequentially on the input
    and performs a residual connection after each module.

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.

    """

    def forward(self, inputs: torch.Tensor, batch: Optional[Batch] = None):
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
        shortcut, outputs = inputs, inputs
        for module in self.values:
            outputs = shortcut + module(outputs, batch=batch)

        return outputs


class ShortcutBlock(Block):
    """
    A block with a 'shortcut' or a 'skip connection'.

    The shortcut tensor can be propagated through the layers of the module or not,
    depending on the value of `propagate_shortcut` argument:
        If `propagate_shortcut` is True, the shortcut tensor is passed through
        each layer of the module.
        If `propagate_shortcut` is False, the shortcut tensor is only used as part of
        the final output dictionary.

    Example usage::
        >>> shortcut = mm.ShortcutBlock(nn.Identity())
        >>> shortcut(torch.ones(1, 1))
            {'shortcut': tensor([[1.]]), 'output': tensor([[1.]])}

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : str, optional
        The name of the module, by default None.
    propagate_shortcut : bool, optional
        If True, propagates the shortcut tensor through the layers of this block, by default False.
    shortcut_name : str, optional
        The name to use for the shortcut tensor, by default "shortcut".
    output_name : str, optional
        The name to use for the output tensor, by default "output".
    """

    def __init__(
        self,
        *module: nn.Module,
        name: Optional[str] = None,
        propagate_shortcut: bool = False,
        shortcut_name: str = "shortcut",
        output_name: str = "output",
    ):
        super().__init__(*module, name=name)
        self.shortcut_name = shortcut_name
        self.output_name = output_name
        self.propagate_shortcut = propagate_shortcut

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Defines the forward propagation of the module.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input tensor or a dictionary of tensors.
        batch : Batch, optional
            A batch of inputs, by default None.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensor as a dictionary.

        Raises
        ------
        RuntimeError
            If the shortcut name is not found in the input dictionary, or
            if the module does not return a tensor or a dictionary with a key 'output_name'.
        """

        if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            if self.shortcut_name not in inputs:
                raise RuntimeError(
                    f"Shortcut name {self.shortcut_name} not found in inputs {inputs}"
                )
            shortcut = inputs[self.shortcut_name]
        else:
            shortcut = inputs

        output = inputs
        for module in self.values:
            if self.propagate_shortcut:
                if torch.jit.isinstance(output, Dict[str, torch.Tensor]):
                    module_output = module(output, batch=batch)
                else:
                    to_pass: Dict[str, torch.Tensor] = {
                        self.shortcut_name: shortcut,
                        self.output_name: torch.jit.annotate(torch.Tensor, output),
                    }

                    module_output = module(to_pass, batch=batch)

                if torch.jit.isinstance(module_output, torch.Tensor):
                    output = module_output
                elif torch.jit.isinstance(module_output, Dict[str, torch.Tensor]):
                    output = module_output[self.output_name]
                else:
                    raise RuntimeError(
                        f"Module {module} must return a tensor or a dict ",
                        f"with key {self.output_name}",
                    )
            else:
                if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]) and torch.jit.isinstance(
                    output, Dict[str, torch.Tensor]
                ):
                    output = output[self.output_name]
                _output = module(output, batch=batch)
                if torch.jit.isinstance(_output, torch.Tensor) or torch.jit.isinstance(
                    _output, Dict[str, torch.Tensor]
                ):
                    output = _output
                else:
                    raise RuntimeError(
                        f"Module {module} must return a tensor or a dict ",
                        f"with key {self.output_name}",
                    )

        to_return = {self.shortcut_name: shortcut}
        if torch.jit.isinstance(output, Dict[str, torch.Tensor]):
            to_return.update(output)
        else:
            to_return[self.output_name] = output

        return to_return


class BatchBlock(Block):
    """
    Class to use for `Batch` creation. We can use this class to create a `Batch` from
        - a tensor or a dictionary of tensors
        - a `Batch` object
        - a tuple of features and targets

    Example usage::
        >>> batch = mm.BatchBlock()(torch.ones(1, 1))
        >>> batch
        Batch(features={"default": tensor([[1.]])})

    """

    def forward(
        self,
        inputs: Union[Batch, TensorOrDict],
        targets: Optional[TensorOrDict] = None,
        sequences: Optional[Sequence] = None,
        batch: Optional[Batch] = None,
    ) -> Batch:
        """
        Perform forward propagation on either a Batch object, or on inputs, targets and sequences
        which are then packed into a Batch.

        Parameters
        ----------
        inputs : Union[Batch, TensorOrDict]
            Either a Batch object or a dictionary of tensors.

        targets : Optional[TensorOrDict], optional
            A dictionary of tensors, by default None

        sequences : Optional[Sequence], optional
            A sequence of tensors, by default None

        batch : Optional[Batch], optional
            A Batch object, by default None

        Returns
        -------
        Batch
            The resulting Batch after forward propagation.
        """
        if torch.jit.isinstance(batch, Batch):
            return self.forward_batch(batch)
        if torch.jit.isinstance(inputs, Batch):
            return self.forward_batch(inputs)

        return self.forward_batch(Batch(inputs, targets, sequences))

    def forward_batch(self, batch: Batch) -> Batch:
        """
        Perform forward propagation on a Batch object.

        For each module in the block, this method performs a forward pass with the
        current output features and the original batch object.
        - If a module returns a Batch object, this becomes the new output.
        - If a module returns a dictionary of tensors, a new Batch object is created
          from this dictionary and the original batch object. The new Batch replaces
          the current output. This is useful when a module only modifies a subset of
          the batch.


        Parameters
        ----------
        batch : Batch
            A Batch object.

        Returns
        -------
        Batch
            The resulting Batch after forward propagation.

        Raises
        ------
        RuntimeError
            When the output of a module is neither a Batch object nor a dictionary of tensors.
        """
        output = batch
        for module in self.values:
            module_out = module(output.features, batch=output)
            if torch.jit.isinstance(module_out, Batch):
                output = module_out
            elif torch.jit.isinstance(module_out, Dict[str, torch.Tensor]):
                output = Batch.from_partial_dict(module_out, batch)
            else:
                raise RuntimeError("Module must return a Batch or a dict of tensors")

        return output


def _validate_n(n: int) -> None:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n < 1:
        raise ValueError("n must be greater than 0")


def repeat(module: nn.Module, n: int = 1, name=None) -> Block:
    """
    Creates a new block by repeating the current block `n` times.
    Each repetition is a deep copy of the current block.

    Parameters
    ----------
    module: nn.Module
        The module to be repeated.
    n : int
        The number of times to repeat the current block.
    name : Optional[str], default = None
        The name for the new block. If None, no name is assigned.

    Returns
    -------
    Block
        The new block created by repeating the current block `n` times.
    """
    _validate_n(n)

    repeats = [module.copy() if hasattr(module, "copy") else deepcopy(module) for _ in range(n - 1)]

    return Block(module, *repeats, name=name)


def repeat_parallel(module: nn.Module, n: int = 1, agg=None) -> ParallelBlock:
    _validate_n(n)

    branches = {"0": module}
    branches.update(
        {str(n): module.copy() if hasattr(module, "copy") else deepcopy(module) for n in range(n)}
    )

    output = ParallelBlock(branches)
    if agg:
        output.append(Block.parse(agg))

    return output


def repeat_parallel_like(module: nn.Module, like: HasKeys, agg=None) -> ParallelBlock:
    branches = {}

    if isinstance(like, Schema):
        keys = like.column_names
    else:
        keys = list(like.keys())

    for i, key in enumerate(keys):
        if i == 0:
            branches[str(key)] = module
        else:
            branches[str(key)] = module.copy() if hasattr(module, "copy") else deepcopy(module)

    output = ParallelBlock(branches)
    if agg:
        output.append(Block.parse(agg))

    return output


def get_pre(module: nn.Module) -> BlockContainer:
    if hasattr(module, "pre"):
        return module.pre

    if isinstance(module, BlockContainer) and module:
        return get_pre(module[0])

    return BlockContainer()


def set_pre(module: nn.Module, pre: BlockContainer):
    if not isinstance(pre, BlockContainer):
        pre = BlockContainer(pre)

    if hasattr(module, "pre"):
        module.pre = pre

    if isinstance(module, BlockContainer):
        return set_pre(module[0], pre)


@schema.input_schema.register(BlockContainer)
def _(module: BlockContainer, input: Schema):
    return schema.input_schema(module[0], input) if module else input


@schema.input_schema.register(ParallelBlock)
def _(module: ParallelBlock, input: Schema):
    if module.pre:
        return schema.input_schema(module.pre)

    out_schema = Schema()
    for branch in module.branches.values():
        out_schema += schema.input_schema(branch, input)

    return out_schema


@schema.output_schema.register(ParallelBlock)
def _(module: ParallelBlock, input: Schema):
    if module.post:
        return schema.output_schema(module.post, input)

    output = Schema()
    for name, branch in module.branches.items():
        branch_schema = schema.output_schema(branch, input)

        if len(branch_schema) == 1 and branch_schema.first.name == "output":
            branch_schema = Schema([branch_schema.first.with_name(name)])

        output += branch_schema

    return output


@schema.output_schema.register(BlockContainer)
def _(module: BlockContainer, input: Schema):
    return schema.output_schema(module[-1], input) if module else input


BlockT = TypeVar("BlockT", bound=BlockContainer)


@schema.select.register(BlockContainer)
def _select_block(container: BlockT, selection: schema.Selection) -> BlockT:
    if isinstance(container, ParallelBlock):
        return _select_parallel_block(container, selection)

    outputs = []

    if not container.values:
        return container.__class__()

    first = container.values[0]
    selected_first = schema.select(first, selection)
    if not selected_first:
        return container.__class__()
    if first == selected_first:
        return container

    outputs.append(selected_first)
    if len(container.values) > 1:
        for module in container.values[1:]:
            try:
                selected_module = schema.select(module, selection)
                outputs.append(selected_module)
            except ValueError:
                selected_module = None
                break

    return container.__class__(*outputs, name=container._name)


SelectT = TypeVar("SelectT", bound=schema.Selectable)
ParallelT = TypeVar("ParallelT", bound=ParallelBlock)


def _select_parallel_block(
    parallel: ParallelT,
    selection: schema.Selection,
) -> ParallelT:
    branches = {}

    pre = parallel.pre
    if pre:
        selected = schema.select(pre, selection)
        if not selected:
            return ParallelBlock()

        pre = selected

    for key, val in parallel.branches.items():
        selected = schema.select(val, selection)
        if selected:
            branches[key] = selected

    if len(branches) == len(parallel.branches):
        post = parallel.post
    else:
        post = BlockContainer()

    replace_kwargs = {"pre": pre, "branches": branches, "post": post}
    if "selection" in inspect.signature(parallel.replace).parameters:
        replace_kwargs["selection"] = selection
    output = parallel.replace(**replace_kwargs)

    return output


def _extract_parallel(main, selection, route, name=None):
    output_branches = {}

    for branch_name, branch in main.branches.items():
        if branch_name in route:
            out = schema.extract.extract(branch, selection, route[branch_name], name=branch_name)
            if out:
                output_branches[branch_name] = out
        else:
            output_branches[branch_name] = branch

    # TODO: What to do with post?
    replace_kwargs = {"branches": output_branches}
    if "selection" in inspect.signature(main.replace).parameters:
        replace_kwargs["selection"] = selection

    return main.replace(**replace_kwargs)


@schema.extract.register(BlockContainer)
def _extract_block(main, selection, route, name=None):
    if isinstance(main, ParallelBlock):
        return _extract_parallel(main, selection, route=route, name=name)

    main_schema = schema.input_schema(main)
    route_schema = schema.input_schema(route)

    if main_schema == route_schema:
        from merlin.models.torch.inputs.select import SelectFeatures

        out_schema = schema.output_schema(main, main_schema)
        if len(out_schema) == 1 and out_schema.first.name == "output":
            out_schema = Schema([out_schema.first.with_name(name)])

        return SelectFeatures(out_schema)

    output = main.__class__()
    for i, module in enumerate(main):
        if i < len(route):
            output.append(schema.extract.extract(module, selection, route[i], name=name))

    return output
