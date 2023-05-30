import torch

from merlin.schema import ColumnSchema, Schema, Tags


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

        if self._handle is None:
            self._handle = self.register_forward_hook(self._post_forward_hook)

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
