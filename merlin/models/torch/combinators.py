from copy import deepcopy
from functools import reduce
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import torch
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from merlin.models.torch.base import (
    TabularBlockMixin,
    _AggModuleWrapper,
    _ModuleWrapper,
    _TabularModuleWrapper,
)
from merlin.models.torch.data import TabularBatch
from merlin.models.torch.transforms.aggregation import SumResidual
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema, Tags


class ParallelBlock(nn.Module, TabularBlockMixin):
    """
    A block that processes inputs in parallel through multiple layers and returns their outputs.

    Parameters
    ----------
    *inputs : Union[nn.Module, Dict[str, nn.Module]]
        Variable length list of PyTorch modules or dictionaries of PyTorch modules.
    pre : Callable, optional
        Preprocessing function to apply on inputs before processing.
    post : Callable, optional
        Postprocessing function to apply on outputs after processing.
    agg : Callable, optional
        Aggregation function to apply on outputs.
    """

    _modules: Dict[str, nn.Module]  # type: ignore[assignment]
    accepts_dict: Final[bool]

    def __init__(
        self,
        *inputs: Union[nn.Module, Dict[str, nn.Module]],
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
        strict: bool = True,
    ):
        super().__init__()

        if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]

        if all(isinstance(x, dict) for x in inputs):
            _parallel_dict = reduce(lambda a, b: dict(a, **b), inputs)
        elif all(isinstance(x, nn.Module) for x in inputs):
            if all(hasattr(m, "name") for m in inputs):
                _parallel_dict = {m.name: m for m in inputs}
            else:
                _parallel_dict = {i: m for i, m in enumerate(inputs)}
        else:
            raise ValueError(f"Invalid input. Got: {inputs}")

        if not strict:
            self.accepts_dict = False
        else:
            # TODO: Handle with pre
            self.accepts_dict = _parallel_check_strict(_parallel_dict, pre=pre, post=post)

        if pre:
            self.pre = _TabularModuleWrapper(pre) if self.accepts_dict else _ModuleWrapper(pre)
        else:
            self.pre = None
        self.post = _TabularModuleWrapper(post) if post else None
        self.agg = _AggModuleWrapper(agg) if agg else None

        # # self.parallel_dict = torch.ModuleDict(_parallel_dict)
        # for key, val in _parallel_dict.items():
        #     self.add_module(str(key), val)

        self.branches = nn.ModuleDict({str(i): m for i, m in _parallel_dict.items()})

        if all(hasattr(m, "schema") for m in _parallel_dict.values()):
            self.schema = reduce(
                lambda a, b: a + b, [m.schema for m in _parallel_dict.values()]
            )  # type: ignore

    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Optional[TabularBatch] = None,
    ):
        """
        Process inputs through the parallel layers.

        Parameters
        ----------
        inputs : Tensor
            Input tensor to process through the parallel layers.
        **kwargs : dict
            Additional keyword arguments for layer processing.

        Returns
        -------
        outputs : dict
            Dictionary containing the outputs of the parallel layers.
        """
        if not self.accepts_dict:
            if not torch.jit.isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["x"]

            x = self.block_prepare_tensor(x, batch=batch)

            outputs = {}

            for name, module in self.branches.items():
                module_inputs = x  # TODO: Add filtering when adding schema
                out = module(module_inputs)

                if isinstance(out, torch.Tensor):
                    out = {name: out}
                elif isinstance(out, tuple):
                    out = {name: out}

                outputs.update(out)

            if torch.jit.isinstance(outputs, TabularBatch):
                outputs = self.block_finalize_batch(outputs, batch=batch)
            else:
                outputs = self.block_finalize(outputs, batch=batch)

            return outputs

        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            raise RuntimeError("Expected a dictionary, but got a tensor instead.")
        x: Dict[str, torch.Tensor] = inputs

        x = self.block_prepare(x, batch=batch)

        outputs = {}

        for name, module in self.branches.items():
            module_inputs = x  # TODO: Add filtering when adding schema
            out = module(module_inputs)

            if isinstance(out, torch.Tensor):
                out = {name: out}
            elif isinstance(out, tuple):
                out = {name: out}

            # TODO: Throw RuntimeError if we overwrite a key
            outputs.update(out)

        if torch.jit.isinstance(outputs, TabularBatch):
            outputs = self.block_finalize_batch(outputs, batch=batch)
        else:
            outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def select_by_name(self, names) -> "ParallelBlock":
        if self.schema is not None and self.schema == self.schema.select_by_name(names):
            return self

        selected_branches = {}
        selected_schemas = Schema()

        for name, branch in self.items():
            branch_has_schema = hasattr(branch, "schema")
            if not branch_has_schema:
                continue
            if not hasattr(branch, "select_by_name"):
                raise AttributeError(
                    f"This ParallelBlock does not support select_by_tag because "
                    f"{branch.__class__} does not support select_by_tag. Consider "
                    "implementing a select_by_name in an extension of "
                    f"{branch.__class__}."
                )
            selected_branch = branch.select_by_name(names)
            if not selected_branch:
                continue
            selected_branches[name] = selected_branch
            selected_schemas += selected_branch.schema

        return ParallelBlock(
            selected_branches,
            post=self.post,
            pre=self.pre,
            aggregation=self.aggregation,
        )

    def select_by_tag(self, tags: Union[str, Tags, List[Union[str, Tags]]]) -> "ParallelBlock":
        """Select branches by tags and return a new ParallelBlock.

        This method will return a ParallelBlock instance with all the branches that
        have at least one feature that matches any of the tags provided.

        For example, this method can be useful when a ParallelBlock has both item and
        user features in a two-tower model or DLRM, and we want to select only the item
        or user features.

        >>> all_inputs = TabularInputBlock(schema)  # TabularInputBlock is a ParallelBlock
        >>> item_inputs = all_inputs.select_by_tag(Tags.ITEM)
        ['continuous', 'embeddings']
        >>> item_inputs.schema["continuous"].column_names
        ['item_recency']
        >>> item_inputs.schema["embeddings"].column_names
        ['item_id', 'item_category', 'item_genres']

        Parameters
        ----------
        tags: str or Tags or List[Union[str, Tags]]
             List of tags that describe which blocks to match

        Returns
        -------
        ParallelBlock
        """
        if self.schema is not None and self.schema == self.schema.select_by_tag(tags):
            return self

        if not isinstance(tags, (list, tuple)):
            tags = [tags]

        selected_branches = {}
        selected_schemas = Schema()

        for name, branch in self.items():
            branch_has_schema = hasattr(branch, "schema")
            if not branch_has_schema:
                continue
            if not hasattr(branch, "select_by_tag"):
                raise AttributeError(
                    f"This ParallelBlock does not support select_by_tag because "
                    f"{branch.__class__} does not support select_by_tag. Consider "
                    "implementing a select_by_tag in an extension of "
                    f"{branch.__class__}."
                )
            selected_branch = branch.select_by_tag(tags)
            if not selected_branch:
                continue
            selected_branches[name] = selected_branch
            selected_schemas += selected_branch.schema

        return ParallelBlock(
            selected_branches,
            post=self.post,
            pre=self.pre,
            aggregation=self.aggregation,
        )

    @_copy_to_script_wrapper
    def items(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.branches.items()

    @_copy_to_script_wrapper
    def keys(self) -> Iterator[str]:
        return self.branches.keys()

    @_copy_to_script_wrapper
    def values(self) -> Iterator[nn.Module]:
        return self.branches.values()

    # @torch.jit.export
    def _first(self):
        for b in self.branches.values():
            return b

    # @property
    @torch.jit.ignore
    def first(self):
        return self._first()

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self.branches)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.branches.values())

    @_copy_to_script_wrapper
    def __getitem__(self, key) -> nn.Module:
        return self.branches[key]

    def __bool__(self) -> bool:
        return bool(self.branches)


class WithShortcut(ParallelBlock):
    """Parallel block for a shortcut connection.

    This block will apply `module` to it's inputs  outputs the following:
    ```python
    {
        module_output_name: module(inputs),
        shortcut_output_name: inputs
    }
    ```

    Parameters:
    -----------
    module : nn.Module
        The input module.
    aggregation : nn.Module or None, optional
        Optional module that aggregates the dictionary into a single tensor.
        Defaults to None.
    post : nn.Module or None, optional
        Optional module that takes in a dict of tensors and outputs a transformed dict of tensors.
        Defaults to None.
    block_outputs_name : str or None, optional
        The name of the output dictionary of the parallel block.
        Defaults to the name of the input module.
    **kwargs : dict
        Additional keyword arguments to be passed to the superclass ParallelBlock.


    """

    def __init__(
        self,
        module: nn.Module,
        *,
        aggregation=None,
        post=None,
        module_output_name="output",
        shortcut_output_name="shortcut",
        **kwargs,
    ):
        super().__init__(
            {module_output_name: module, shortcut_output_name: nn.Identity()},
            post=post,
            aggregation=aggregation,
            **kwargs,
        )


class ResidualBlock(WithShortcut):
    """
    Residual block for a shortcut connection with a sum operation and optional activation.

    Parameters
    ----------
    module : nn.Module
        The input module.
    activation : Union[Callable[[torch.Tensor], torch.Tensor], str], optional
        Activation function to be applied after the sum operation.
        It can be a callable or a string representing a standard activation function.
        Defaults to None.
    post : nn.Module or None, optional
        Optional module that takes in a dict of tensors and outputs a transformed dict of tensors.
        Defaults to None.
    **kwargs : dict
        Additional keyword arguments to be passed to the superclass WithShortcut.

    Examples
    --------
    >>> linear = nn.Linear(5, 3)
    >>> residual_block = ResidualBlock(linear, activation=nn.ReLU())

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        activation: Optional[nn.Module] = None,
        post=None,
        **kwargs,
    ):
        super().__init__(
            module,
            post=post,
            aggregation=SumResidual(activation=activation),
            **kwargs,
        )


class SequentialBlock(nn.Sequential):
    """A sequential container.
    This class extends PyTorch's Sequential class with additional functionalities.

    Parameters
    ----------
    *args : nn.Module
        Variable-length list of modules to be added sequentially.
    pre : nn.Module, optional
        An optional module to be applied before the input is passed to the sequential modules.
    post : nn.Module, optional
        An optional module to be applied after the output is obtained from the sequential modules.

    Methods
    -------
    append_with_shortcut(module, post=None, aggregation=None)
        Appends a module with a shortcut connection to the sequential block.
    append_with_residual(module, activation=None, **kwargs)
        Appends a module with a residual connection to the sequential block.
    append_branch(*branches, post=None, aggregation=None, **kwargs)
        Appends a branch of parallel modules to the sequential block.
    repeat(num=1, copies=True)
        Repeats the sequential block a specified number of times.
    repeat_in_parallel(num=1, prefix=None, names=None, post=None, aggregation=None,
        copies=True, shortcut=False, **kwargs)
        Repeats the sequential block in parallel a specified number of times.
    """

    def __init__(self, *args, pre=None, post=None):
        super().__init__(*args)
        # self.pre = register_pre_hook(self, pre) if pre else None
        # self.post = register_post_hook(self, post) if post else None

    def append_with_shortcut(
        self,
        module: nn.Module,
        *,
        post=None,
        aggregation=None,
    ) -> "SequentialBlock":
        """Appends a module with a shortcut connection to the sequential block.

        Parameters
        ----------
        module : nn.Module
            The module to be added with a shortcut connection.
        post : nn.Module, optional
            An optional module that takes in a dict of tensors and outputs
            a transformed dict of tensors.
        aggregation : nn.Module, optional
            An optional module that aggregates the dictionary output of the
            parallel block into a single tensor.

        Returns
        -------
        SequentialBlock
            The updated sequential block with the appended module.
        """
        return self.append(WithShortcut(module, post=post, aggregation=aggregation))

    def append_with_residual(
        self, module: nn.Module, *, activation=None, **kwargs
    ) -> "SequentialBlock":
        """Appends a module with a residual connection to the sequential block.

        Parameters
        ----------
        module : nn.Module
            The module to be added with a residual connection.
        activation : callable or str, optional
            The activation function to be applied after the residual connection.

        Returns
        -------
        SequentialBlock
            The updated sequential block with the appended module.
        """
        return self.append(ResidualBlock(module, activation=activation, **kwargs))

    def append_branch(
        self,
        *branches: Union[nn.Module, Dict[str, nn.Module]],
        post=None,
        aggregation=None,
        **kwargs,
    ) -> "SequentialBlock":
        """Appends a branch of parallel modules to the sequential block.

        Parameters
        ----------
        *branches : nn.Module
            Variable-length list of modules to be added in parallel.
        post : nn.Module, optional
            An optional module to be applied after the output is obtained
            from the parallel branches.
        aggregation : nn.Module, optional
            An optional module that aggregates the outputs from the parallel
            branches into a single tensor.

        Returns
        -------
        SequentialBlock
            The updated sequential block with the appended branch of parallel modules.
        """
        return self.append(ParallelBlock(*branches, post=post, aggregation=aggregation, **kwargs))

    def repeat(self, num: int = 1, copies=True) -> "SequentialBlock":
        """
        Repeats the sequential block a specified number of times.

        Parameters
        ----------
        num : int, optional
            The number of times to repeat the sequential block. Default is 1.
        copies : bool, optional
            Whether to create deep copies of the modules for each repetition. Default is True.

        Returns
        -------
        SequentialBlock
            The repeated sequential block.
        """
        self_modules = list(self._modules.values())
        output_modules = [*self_modules]
        for _ in range(num):
            if copies:
                repeated = deepcopy(self_modules)
                for m in repeated:
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
            else:
                repeated = self_modules
            output_modules.extend(repeated)

        return SequentialBlock(*output_modules, pre=self.pre, post=self.post)

    def repeat_in_parallel(
        self,
        num: int = 1,
        prefix=None,
        names=None,
        post=None,
        aggregation=None,
        copies=True,
        shortcut=False,
        **kwargs,
    ) -> "ParallelBlock":
        """
        Repeats the sequential block in parallel a specified number of times.

        Parameters
        ----------
        num : int, optional
            The number of times to repeat the sequential block in parallel. Default is 1.
        prefix : str, optional
            A string prefix to be used for naming the repeated branches if names are not provided.
        names : list of str, optional
            A list of names for the repeated branches.
            If not provided, the branches will be named using the prefix and an integer index.
        post : nn.Module, optional
            An optional module to be applied after the output is obtained
            from the parallel branches.
        aggregation : nn.Module, optional
            An optional module that aggregates the outputs from the
            parallel branches into a single tensor.
        copies : bool, optional
            Whether to create deep copies of the modules for each repetition. Default is True.
        shortcut : bool, optional
            Whether to include a shortcut connection in the parallel branches. Default is False.

        Returns
        -------
        ParallelBlock
            The parallel block containing the repeated sequential blocks.
        """
        repeated = {}
        iterator = names if names else range(num)
        self_modules = list(self._modules.values())

        if not names and prefix:
            iterator = [f"{prefix}{num}" for num in iterator]
        for name in iterator:
            branch = self
            if copies:
                branch_modules = deepcopy(self_modules)
                for m in branch_modules:
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                branch = SequentialBlock(*branch_modules, pre=self.pre, post=self.post)
            repeated[str(name)] = branch

        if shortcut:
            repeated["shortcut"] = nn.Identity()

        return ParallelBlock(repeated, post=post, aggregation=aggregation, **kwargs)


def _parallel_check_strict(
    parallel: Dict[str, nn.Module],
    pre: Optional[nn.Module] = None,
    post: Optional[nn.Module] = None,
) -> bool:
    pre_input_type, pre_output_type = None, None

    if pre:
        pre_input_type, pre_output_type = module_utils.torchscript_io_types(pre)

    parallel_input_types = {}
    parallel_output_types = {}

    for name, module in parallel.items():
        input_type, output_type = module_utils.torchscript_io_types(module)
        parallel_input_types[name] = input_type
        parallel_output_types[name] = output_type

        if pre and pre_output_type != input_type:
            raise ValueError(
                f"Input type mismatch between pre module and parallel module {name}: {pre_output_type} != {input_type}. "
                "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
                "Consider annotating one of the provided modules."
            )

    first_parallel_input_type = next(iter(parallel_input_types.values()))
    if not all(i_type == first_parallel_input_type for i_type in parallel_input_types.values()):
        raise ValueError(
            f"Input type mismatch among parallel modules: {parallel_input_types}. "
            "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
            "Consider annotating one of the provided modules."
        )

    if post:
        parallel_out = f"Dict[str, {first_parallel_input_type}]"
        post_input_type, _ = module_utils.torchscript_io_types(post)

        if parallel_out != post_input_type:
            raise ValueError(
                f"Output type mismatch between parallel modules and post module: {parallel_output_types} != {post_input_type}. "
                "If the input argument in forward is not annotated, TorchScript assumes it's of type Tensor. "
                "Consider annotating one of the provided modules."
            )

    inp_type = pre_input_type if pre else first_parallel_input_type

    return inp_type == "Dict[str, Tensor]"
