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

import types
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import nn

from merlin.dispatch.lazy import LazyDispatcher
from merlin.models.torch.batch import Batch
from merlin.models.torch.utils.module_utils import check_batch_arg
from merlin.schema import ColumnSchema, Schema, Tags, TagSet

Selection = Union[Schema, ColumnSchema, Callable[[Schema], Schema], Tags, TagSet, List[Tags]]
ToSelectT = TypeVar("ToSelectT")
NAMESPACE_TAGS = [Tags.CONTEXT, Tags.USER, Tags.ITEM, Tags.SESSION]


class LazySchemaModuleMixin:
    _initialized_from_schema = False

    def initialize_from_schema(self, schema):
        if self._initialized_from_schema:
            raise RuntimeError("Already initialized this module from a schema")
        self._initialized_from_schema = True


def default_tag_propagation(inputs: Schema, outputs: Schema):
    if inputs:
        namespaces = []
        for tag in NAMESPACE_TAGS:
            namespace = inputs.select_by_tag(tag)
            if namespace and len(namespace) == len(inputs):
                namespaces.append(tag)

        if namespaces:
            to_return = Schema()
            for col in outputs:
                to_return[col.name] = col.with_tags(namespaces)

            return to_return

    return outputs


class _LazyDispatchPyTorch(LazyDispatcher):
    def __init__(self, func_or_name):
        super().__init__(func_or_name)
        self._tensor_registry = {}

    def register_tensor(self, tensor_type, func=None):
        if func is None:
            return lambda f: self.register_tensor(tensor_type, f)
        self._tensor_registry[tensor_type] = func

        return func

    def tensors(self, inputs):
        for tensor_type, func in self._tensor_registry.items():
            if torch.jit.isinstance(inputs, tensor_type):
                return func(inputs)

        raise NotImplementedError(f"Schema not registered for {type(inputs)}")

    def get_schema(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor], Schema]) -> Schema:
        if isinstance(inputs, Schema):
            return inputs

        return self.tensors(inputs)


class _InputSchemaDispatch(_LazyDispatchPyTorch):
    def __call__(self, module: nn.Module, inputs: Optional[Schema] = None) -> Schema:
        if hasattr(module, "input_schema"):
            output = module.input_schema
            if isinstance(output, types.MethodType):
                return output()

            return output

        input_schemas = getattr(module, "__input_schemas", None)
        if input_schemas:
            if len(input_schemas) == 1:
                return input_schemas[0]

            if inputs is None:
                raise ValueError("Must provide inputs to get output schema")

            # TODO: Fix this properly
            i = module.__input_schemas.index(inputs)

            return input_schemas[i]

        try:
            return super().__call__(module, inputs)
        except NotImplementedError:
            raise ValueError(
                f"Could not get output schema of {module} " "please call `mm.schema.trace` first."
            )

    def trace(
        self, module: nn.Module, inputs: Union[torch.Tensor, Dict[str, torch.Tensor], Schema]
    ) -> Schema:
        inputs_schema = self.get_schema(inputs)

        try:
            return super().__call__(module, inputs_schema)
        except NotImplementedError:
            return inputs_schema


class _OutputSchemaDispatch(_LazyDispatchPyTorch):
    def register(self, cls, func=None, tag_propagation_func=default_tag_propagation):
        if tag_propagation_func:

            def _func(module: nn.Module, input: Schema) -> Schema:
                return tag_propagation_func(input, func(module, input))

            if func is None:
                return lambda f: self.register(cls, f, tag_propagation_func=tag_propagation_func)

            return self.dispatcher.register(cls, func=_func)

        return super().register(cls, func=func)

    def __call__(self, module: nn.Module, inputs: Optional[Schema] = None) -> Schema:
        try:
            _inputs = input_schema(module)
            inputs = _inputs
        except ValueError:
            pass

        if hasattr(module, "output_schema"):
            output = module.output_schema
            if isinstance(output, types.MethodType):
                return default_tag_propagation(inputs, output())

            return output

        output_schemas = getattr(module, "__output_schemas", None)
        if output_schemas:
            if len(output_schemas) == 1:
                return output_schemas[0]

            if inputs is None:
                raise ValueError("Must provide inputs to get output schema")

            # TODO: Fix this properly
            i = module.__input_schemas.index(inputs)

            return output_schemas[i]

        try:
            return super().__call__(module, inputs)
        except NotImplementedError:
            raise ValueError(
                f"Could not get output schema of {module} " "please call `mm.schema.trace` first."
            )

    def trace(
        self,
        module: nn.Module,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], Schema],
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor], Schema],
    ) -> Schema:
        _input_schema = input_schema.get_schema(inputs)
        _output_schema = self.get_schema(outputs)

        try:
            output = super().__call__(module, _input_schema)

            if output == _input_schema:
                return _output_schema

            return output
        except NotImplementedError:
            return _output_schema


class _SelectDispatch(LazyDispatcher):
    def __call__(self, to_select: ToSelectT, selection: Selection) -> ToSelectT:
        if hasattr(to_select, "select") and not isinstance(to_select, Schema):
            output = to_select.select(selection)
        else:
            output = super().__call__(to_select, selection)

        return output

    def dispatched(self, to_select: ToSelectT, selection: Selection) -> ToSelectT:
        return super().__call__(to_select, selection)


class _ExtractDispatch(LazyDispatcher):
    def __call__(self, module: nn.Module, selection: Selection) -> Tuple[nn.Module, nn.Module]:
        if hasattr(module, "extract"):
            return module.extract(selection)

        extraction = select(module, selection)
        module_with_extraction = self.extract(module, selection, extraction)

        return module_with_extraction, extraction

    def extract(self, module: nn.Module, selection: Selection, route: nn.Module, name=None):
        fn = self.dispatch(module)
        return fn(module, selection, route, name=name)


input_schema = _InputSchemaDispatch("input_schema")
output_schema = _OutputSchemaDispatch("output_schema")
select = _SelectDispatch("selection")
extract = _ExtractDispatch("extract")


def trace(module: nn.Module, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs):
    """
    Traces schemas for a given PyTorch module and all it's children.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to trace.
    inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
        The input to the module's forward method, it could be either a single Tensor
        or a dictionary of Tensors.
    **kwargs
        Arbitrary keyword arguments to pass to the forward-pass.

    Returns
    -------
    output : torch.Tensor
        The output tensor of the module's forward pass.
    """

    hooks = []
    batch = kwargs.get("batch")

    def _hook(mod: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        if not hasattr(mod, "__input_schemas"):
            mod.__input_schemas = ()
            mod.__output_schemas = ()
            mod.__feature_schema = None

        _input_schema = input_schema.trace(mod, inputs[0])
        initialize_from_schema(mod, _input_schema)

        if _input_schema not in mod.__input_schemas:
            mod.__input_schemas += (_input_schema,)
            mod.__output_schemas += (output_schema.trace(mod, _input_schema, outputs),)

        if not isinstance(mod, torch.jit.TracedModule):
            accepts_batch, requires_batch = check_batch_arg(mod)

            if requires_batch and batch is None:
                raise ValueError(
                    f"Found module f{mod} that requires a batch. "
                    "`trace` was called without providing a batch. "
                    "Please provide a batch argument to `trace`. "
                )
            if accepts_batch and hasattr(mod, "compute_feature_schema"):
                feature_schema = input_schema.tensors(batch.features)
                required_feature_schema = mod.compute_feature_schema(feature_schema)
                mod.__feature_schema = required_feature_schema
            elif requires_batch:
                required_feature_schema = input_schema.tensors(batch.features)
                mod.__feature_schema = required_feature_schema

    def add_hook(m):
        custom_modules = list(output_schema.dispatcher.registry.keys())
        if m and isinstance(m, tuple(custom_modules[1:])):
            return

        hooks.append(m.register_forward_hook(_hook))

    module.apply(add_hook)
    module_out = module(inputs, **kwargs)

    for hook in hooks:
        hook.remove()

    return module_out


def feature_schema(module: nn.Module) -> Schema:
    """Extract the feature schema from a PyTorch Module.

    This function operates by applying the `get_feature_schema` method
    to each submodule within the provided PyTorch Module. It checks
    if the submodule has a `feature_schema` attribute and, if so,
    adds this to the output schema.

    Parameters
    ----------
    module : nn.Module
        The PyTorch Module from which to extract the feature schema.

    Returns
    -------
    Schema
        The feature schema extracted from the PyTorch Module.

    """

    feature_schema = Schema()

    def get_feature_schema(module):
        nonlocal feature_schema
        if hasattr(module, "__feature_schema"):
            feature_schema += module.__feature_schema

    module.apply(get_feature_schema)

    return feature_schema


def target_schema(module: nn.Module) -> Schema:
    """
    Extract the target schema from a PyTorch Module.

    This function operates by applying the `get_target_schema` method
    to each submodule within the provided PyTorch Module. It checks
    if the submodule has a `target_schema` attribute and, if so,
    adds this to the output schema.

    Parameters
    ----------
    module : nn.Module
        The PyTorch Module from which to extract the target schema.

    Returns
    -------
    Schema
        The target schema extracted from the PyTorch Module.

    """

    target_schema = Schema()

    def get_target_schema(module):
        nonlocal target_schema
        if hasattr(module, "target_schema"):
            target_schema += module.target_schema

    module.apply(get_target_schema)

    return target_schema


def initialize_from_schema(module: nn.Module, schema: Schema):
    """
    Set up a schema for a given module.

    Parameters
    ----------
    module : nn.Module
        The module for which to set up the schema.
    schema : Schema
        The schema to set up.
    """

    from merlin.models.torch.block import BlockContainer, ParallelBlock

    if hasattr(module, "initialize_from_schema") and not getattr(
        module, "_initialized_from_schema", False
    ):
        module.initialize_from_schema(schema)
        module._initialized_from_schema = True

    elif isinstance(module, ParallelBlock):
        for branch in module.branches.values():
            initialize_from_schema(branch, schema)

    elif isinstance(module, BlockContainer) and module:
        initialize_from_schema(module[0], schema)


@select.register(Schema)
def select_schema(schema: Schema, selection: Selection) -> Schema:
    """
    Select a subset of a schema based on the selection criteria.

    Parameters
    ----------
    schema : Schema
        The original schema to select from.
    selection : Selection
        The selection criteria. Can be a Schema, ColumnSchema,
        a callable that returns a Schema, or Tags.

    Returns
    -------
    Schema
        The selected subset of the schema.

    Raises
    ------
    ValueError
        If the selection criteria is not a valid type.
    """
    if not isinstance(schema, Schema):
        raise ValueError(f"Schema {schema} is not valid")

    if isinstance(selection, Schema):
        selected = selection
    elif isinstance(selection, ColumnSchema):
        if selection.name not in schema.column_names:
            return Schema()
        selected = Schema([schema[selection.name]])
    elif callable(selection):
        selected = selection(schema)
    elif isinstance(selection, (Tags, TagSet)):
        selected = schema.select_by_tag(selection)
    elif isinstance(selection, str):
        if selection == "*":
            return schema

        selected = Schema([schema[selection]])
    elif isinstance(selection, (list, tuple)):
        if all(isinstance(s, str) for s in selection):
            selected = Schema([schema[s] for s in selection])
        elif all(isinstance(s, ColumnSchema) for s in selection):
            selected = Schema([schema[s.name] for s in selection])
        elif all(isinstance(s, (Tags, TagSet)) for s in selection):
            selected = schema.select_by_tag(selection)
        else:
            raise ValueError(f"Selection {selection} is not valid")
    else:
        raise ValueError(f"Selection {selection} is not valid")

    return selected


def select_union(*selections: Selection) -> Selection:
    """
    Combine selections into a single selection.

    This function returns a new function `combined_select` that, when called,
    will perform the union operation on all the input selections.

    Parameters
    ----------
    *selections : Selection
        Variable length argument list of Selection instances.

    Returns
    -------
    Selection
        A function that takes a Schema as input and returns a Schema which
        is the union of all selections.
    """

    def combined_select(schema: Schema) -> Schema:
        output = Schema()
        for s in selections:
            output += select(schema, s)

        return output

    return combined_select


def selection_name(selection: Selection) -> str:
    """
    Get the name of the selection.

    Parameters
    ----------
    selection : Selection
        The selection criteria. Can be a Schema, ColumnSchema, a callable
        that returns a Schema, or Tags.

    Returns
    -------
    str
        The name of the selection.

    Raises
    ------
    ValueError
        If the selection criteria is not a valid type.
    """
    if isinstance(selection, ColumnSchema):
        return selection.name
    elif isinstance(selection, Tags):
        return selection.value
    elif isinstance(selection, Schema):
        return "_".join(selection.column_names)
    elif callable(selection):
        return selection.__name__

    raise ValueError(f"Selection {selection} is not valid")


class Selectable:
    """
    A mixin to allow to be selectable by schema.
    """

    def initialize_from_schema(self, schema: Schema):
        """
        Setup the schema for this selectable.

        Parameters
        ----------
        schema : Schema
            The schema to setup.

        Returns
        -------
        Selectable
            Self for chaining.
        """
        self.schema = schema

        return self

    def select(self, selection: Selection) -> "Selectable":
        """
        Select a subset of the schema.

        This method should be overridden by subclasses.

        Parameters
        ----------
        selection : Selection
            The selection criteria. Can be a Schema, ColumnSchema, a callable that
            returns a Schema, or Tags.

        Returns
        -------
        Selectable
            A new selectable with the selected subset of the schema.

        Raises
        ------
        NotImplementedError
            If this method is not overridden by a subclass.
        """
        raise NotImplementedError()


@output_schema.register_tensor(torch.Tensor)
def _tensor_to_schema(input, name="output"):
    if input is None:
        return Schema([ColumnSchema(name)])

    kwargs = dict(dims=input.shape[1:], dtype=input.dtype)

    if len(input.shape) > 1 and input.dtype != torch.int32:
        kwargs["tags"] = [Tags.EMBEDDING]

    return Schema([ColumnSchema(name, **kwargs)])


@input_schema.register_tensor(torch.Tensor)
def _(input):
    return _tensor_to_schema(input, "input")


@input_schema.register_tensor(Dict[str, torch.Tensor])
@output_schema.register_tensor(Dict[str, torch.Tensor])
def _(input):
    output = Schema()
    for k, v in sorted(input.items()):
        if k.endswith("__offsets"):
            name = k[: -len("__offsets")]
            kwargs = dict(dtype=input[f"{name}__values"].dtype)
            output += Schema([ColumnSchema(name, **kwargs)])
        elif k.endswith("__values"):
            continue
        else:
            output += _tensor_to_schema(v, k)

    return output


@input_schema.register_tensor(Tuple[torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, Optional[torch.Tensor]])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
@input_schema.register_tensor(
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)
@output_schema.register_tensor(
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)
@input_schema.register_tensor(
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)
@output_schema.register_tensor(
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)
@input_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@output_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@input_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@output_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@input_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@output_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@input_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
@output_schema.register_tensor(
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
)
def _(input):
    output = Schema()

    for i, v in enumerate(input):
        output += _tensor_to_schema(v, str(i))

    return output


@output_schema.register_tensor(Batch)
def _(input):
    schema = Schema()
    schema += output_schema.tensors(input.features)
    schema += output_schema.tensors(input.targets)

    return schema
