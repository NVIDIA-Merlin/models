import types
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from merlin.dispatch.lazy import LazyDispatcher
from merlin.schema import ColumnSchema, Schema, Tags


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
                f"Could not get output schema of {module} " "please call mm.trace_schema first."
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
    def __call__(self, module: nn.Module, inputs: Optional[Schema] = None) -> Schema:
        if hasattr(module, "output_schema"):
            output = module.output_schema
            if isinstance(output, types.MethodType):
                return output()

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
                f"Could not get output schema of {module} " "please call mm.trace_schema first."
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
            return super().__call__(module, _input_schema)
        except NotImplementedError:
            return _output_schema


input_schema = _InputSchemaDispatch("input_schema")
output_schema = _OutputSchemaDispatch("output_schema")


@output_schema.register_tensor(torch.Tensor)
def _tensor_to_schema(input, name="output"):
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
        output += _tensor_to_schema(v, k)

    return output


@input_schema.register_tensor(Tuple[torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
@input_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
@output_schema.register_tensor(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
def _(input):
    output = Schema()

    for i, v in enumerate(input):
        output += _tensor_to_schema(v, str(i))

    return output


def trace_schema(module: nn.Module, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs):
    hooks = []

    def _hook(mod: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        if not hasattr(mod, "__input_schemas"):
            mod.__input_schemas = ()
            mod.__output_schemas = ()

        _input_schema = input_schema.trace(mod, input[0])
        if _input_schema not in mod.__input_schemas:
            mod.__input_schemas += (_input_schema,)
            mod.__output_schemas += (output_schema.trace(mod, _input_schema, output),)

    def add_hook(m):
        custom_modules = list(output_schema.dispatcher.registry.keys())
        if len(custom_modules) > 0 and isinstance(m, tuple(custom_modules[1:])):
            return

        hooks.append(m.register_forward_hook(_hook))

    module.apply(add_hook)
    output = module(inputs, **kwargs)

    for hook in hooks:
        hook.remove()

    return output


class SchemaTrackingMixin:
    """
    A mixin class for PyTorch modules to track the output shapes and dtypes
    of the forward pass. This is used in order to automatically generate
    the output-schema.

    It registers a hook to capture this information and
    provides methods to access the output schema, as well as to set the module
    in training or evaluation mode.
    """

    def __init__(self):
        super().__init__()
        self._register_schema_tracking_hook()

    def _post_forward_hook(self, module, input, output):
        """Hook function to be called after the forward pass of the module.

        Parameters
        ----------
        module : torch.nn.Module
            The module for which the forward pass was called.
        input : tuple
            The input arguments passed to the forward method.
        output : torch.Tensor or dict
            The output of the forward method.
        """
        if not module._forward_called:
            if isinstance(output, dict):
                for key, value in output.items():
                    module._output_shapes[key] = value.shape
                    module._output_dtypes[key] = value.dtype
            else:
                module._output_shapes["output"] = output.shape
                module._output_dtypes["output"] = output.dtype
            module._forward_called = True
            module._handle.remove()

    def _register_schema_tracking_hook(self):
        """
        Register the post forward hook to the module.
        """
        self._forward_called = False
        self._handle = None
        self._output_shapes = {}
        self._output_dtypes = {}
        self._output_schema = None

        if self._handle is None:
            self._handle = self.register_forward_hook(self._post_forward_hook)

    @torch.jit.ignore
    def output_schema(self) -> Schema:
        """Get the output schema of the module.

        Returns
        -------
        Schema
            The output schema of the module.

        Raises
        ------
        RuntimeError
            If forward() has not been called before calling this method.
        """

        if getattr(self, "_output_schema", None):
            return self._output_schema

        if not hasattr(self, "_output_shapes"):
            raise RuntimeError(
                "Schema-tracking hook not registered, use `_register_schema_tracking_hook`."
            )

        if not self._forward_called:
            raise RuntimeError("forward() must be called before output_schema() can be called.")

        columns = []

        for name, shape in self._output_shapes.items():
            dtype = self._output_dtypes[name]
            dims = (None,) + tuple(shape)
            tags = None

            if len(shape) > 1 and dtype != torch.int32:
                tags = [Tags.EMBEDDING]

            columns.append(ColumnSchema(name, dims=dims, tags=tags, dtype=dtype))

        return Schema(columns)

    def train(self, mode=True):
        self._register_schema_tracking_hook()
        return super().train(mode)

    def eval(self):
        self._register_schema_tracking_hook()
        return super().eval()


def get_input_schema(module) -> Schema:
    """
    Get the input schema of a module.

    Parameters
    ----------
    module : object
        The module from which to retrieve the input schema.

    Returns
    -------
    Schema
        The input schema of the module. If the module does not have
        a specific input schema, a new Schema instance is created
        and updated with the schemas of the sub-items (if any) in the module.
    """

    if hasattr(module, "input_schema"):
        return module.input_schema()
    if hasattr(module, "schema"):
        return module.schema

    schema = Schema()
    if hasattr(module, "items"):
        for value in module.values():
            schema += get_input_schema(value)

    else:
        # check if iterable
        try:
            return get_input_schema(module[0])
        except TypeError:
            pass

    return schema


def get_output_schema(module) -> Schema:
    """
    Get the output schema of a module.

    Parameters
    ----------
    module : object
        The module from which to retrieve the output schema.

    Returns
    -------
    Schema
        The output schema of the module. If the module does not have
        a specific output schema, a new Schema instance is created
        and updated with the schemas of the sub-items (if any) in the module.
    """
    if hasattr(module, "output_schema"):
        return module.output_schema()

    schema = Schema()
    if hasattr(module, "items"):
        for value in module.values():
            schema += get_output_schema(value)

    else:
        # check if iterable
        try:
            return get_output_schema(module[-1])
        except TypeError:
            pass

    return schema


def setup_schema(module: nn.Module, schema: Schema):
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

    if hasattr(module, "setup_schema"):
        module.setup_schema(schema)

    elif isinstance(module, ParallelBlock):
        for branch in module.branches.values():
            setup_schema(branch, schema)

    elif isinstance(module, BlockContainer) and module:
        setup_schema(module[0], schema)
