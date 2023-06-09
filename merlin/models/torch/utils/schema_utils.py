from typing import Dict, Tuple, Union

import torch
from torch import nn

from merlin.dispatch.lazy import LazyDispatcher
from merlin.schema import ColumnSchema, Schema, Tags


class _InputSchemaDispatch(LazyDispatcher):
    def __call__(self, module: nn.Module) -> Schema:
        return super().__call__(module)


class _OutputSchemaDispatch(LazyDispatcher):
    def __call__(self, module: nn.Module, input_schema: Schema) -> Schema:
        try:
            return super().__call__(module, input_schema)
        except NotImplementedError:
            if not hasattr(module, "_tracked_schema"):
                raise RuntimeError("Schema-tracking hook not registered for module")

            ...


_input_schema = _InputSchemaDispatch("input_schema")
_output_schema = _OutputSchemaDispatch("output_schema")


def trace_schema(module: nn.Module, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    hooks = []

    def _hook(mod: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        if not hasattr(mod, "__traced_schema"):
            mod.__traced_schema = {}

        # Collect all inputs information
        # TODO: Make this more generic
        if isinstance(input[0], torch.Tensor):
            input_schema = Schema(
                ColumnSchema("input", dims=input[0].shape[1:], dtype=input[0].dtype)
            )
        elif isinstance(input[0], dict):
            input_schema = Schema(
                [
                    ColumnSchema(k, dims=v.shape[1:], dtype=v.dtype)
                    for k, v in sorted(input[0].items())
                ]
            )
        elif isinstance(input[0], (tuple, list)):
            input_schema = Schema(
                [
                    ColumnSchema(str(i), dims=v.shape[1:], dtype=v.dtype)
                    for i, v in enumerate(input[0])
                ]
            )
        else:
            raise ValueError(f"Unhandled input type: {type(input[0])}")

        # Convert to string to use as a dictionary key
        key = str(input_schema)

        # Save output shape and dtype
        out_shape = output.data.shape
        out_dtype = output.dtype

        mod.__traced_schema[key] = {
            "output_shape": out_shape,
            "output_dtype": out_dtype,
        }

    for m in module.modules():
        custom_modules = list(_output_schema.dispatcher.registry.keys())
        if len(custom_modules) > 0 and isinstance(m, tuple(custom_modules[1:])):
            continue
        if m.__class__ in (nn.ModuleList, nn.ModuleDict):
            continue

        hooks.append(m.register_forward_hook(_hook))

    output = module(inputs)

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


def input_schema(module) -> Schema:
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
            schema += input_schema(value)

    else:
        # check if iterable
        try:
            return input_schema(module[0])
        except TypeError:
            pass

    return schema


def output_schema(module) -> Schema:
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
            schema += output_schema(value)

    else:
        # check if iterable
        try:
            return output_schema(module[-1])
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
