import inspect
from functools import lru_cache
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.typing import TabularData
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema

_FEATURE_PREFIX = "__buffer_feature"
_TARGET_PREFIX = "__buffer_target"


def sample_batch(
    dataset_or_loader: Union[Dataset, Loader],
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = False,
    include_targets: Optional[bool] = True,
) -> TabularData:
    """Util function to generate a batch of input tensors from a merlin.io.Dataset instance

    Parameters
    ----------
    data: merlin.io.dataset
        A Dataset object.
    batch_size: int
        Number of samples to return.
    shuffle: bool
        Whether to sample a random batch or not, by default False.
    include_targets: bool
        Whether to include the targets in the returned batch, by default True.

    Returns:
    -------
    batch: Dict[torch.Tensor]
        dictionary of input tensors.
    """

    if isinstance(dataset_or_loader, Dataset):
        if not batch_size:
            raise ValueError("Either use 'Loader' or specify 'batch_size'")
        loader = Loader(dataset_or_loader, batch_size=batch_size, shuffle=shuffle)
    else:
        loader = dataset_or_loader

    batch = loader.peek()
    # batch could be of type Prediction, so we can't unpack directly
    inputs, targets = batch[0], batch[1]

    if not include_targets:
        return inputs

    return inputs, targets


class _FeatureReceiverHook(nn.Module):
    KEY_NAME = "features"
    PROPERTY_NAME = "__feature_receiver"

    def __init__(self, schema: Schema):
        super().__init__()
        self.schema = schema

    def forward(self, module, inputs, kwargs):
        if self.KEY_NAME in kwargs:
            return inputs, kwargs

        args_count, kwargs_count, has_args, has_kwargs = _count_function_params(module.forward)
        if not (has_args and has_kwargs) and len(inputs) == args_count + kwargs_count:
            return inputs, kwargs

        kwargs[self.KEY_NAME] = _get_features(module)

        return inputs, kwargs

    @classmethod
    def propagate(cls, module, features):
        _upsert_buffers(module, features, "feature")
        _upsert_buffer(module, cls.PROPERTY_NAME, torch.tensor(True))

    @classmethod
    def needs_propagation(cls, module: nn.Module) -> bool:
        return hasattr(module, cls.PROPERTY_NAME)


class _TargetReceiverHook(nn.Module):
    KEY_NAME = "targets"
    PROPERTY_NAME = "__target_receiver"

    def forward(self, module, inputs, kwargs):
        if self.KEY_NAME in kwargs:
            return inputs, kwargs

        args_count, kwargs_count, has_args, has_kwargs = _count_function_params(module.forward)

        if not (has_args and has_kwargs):
            if len(inputs) == 1 and isinstance(inputs[0], tuple):
                if len(inputs[0]) == args_count + kwargs_count:
                    return inputs[0], kwargs
            if len(inputs) == args_count + kwargs_count:
                return inputs, kwargs

        maybe_targets = _get_targets(module)
        if maybe_targets is not None:
            kwargs[self.KEY_NAME] = maybe_targets

        return inputs, kwargs

    @classmethod
    def propagate(cls, module, targets):
        _upsert_buffers(module, targets, "target")
        _upsert_buffer(module, cls.PROPERTY_NAME, torch.tensor(True))

    @classmethod
    def needs_propagation(cls, module: nn.Module) -> bool:
        return hasattr(module, cls.PROPERTY_NAME)


class _DataPropagationHook(nn.Module):
    """A data propagation hook for PyTorch modules.

    This hook allows you to propagate features and/or targets through
    the children of a model during the forward pass.
    """

    _FEATURE_HOOK = _FeatureReceiverHook
    _TARGET_HOOK = _TargetReceiverHook

    def __init__(self):
        super().__init__()

    def forward(self, model, inputs, kwargs, outputs=None):
        if outputs is None:
            return self.pre_forward(model, inputs, kwargs)

        return self.post_forward(model, outputs)

    def pre_forward(self, model, inputs, kwargs):
        """Forward pass for the DataPropagationHook.

        This function is called when the hook is executed
        during the forward pass of the parent module.

        Args:
            model (nn.Module): The parent module for which the hook is registered.
            inputs: The inputs to the parent module.
            kwargs: Additional keyword arguments to the parent module.
                This is used to retrieve the targets.

        Returns:
            Tuple: The original inputs and empty dict for kwargs.
        """
        targets = kwargs.get("targets", None)

        for child in module_utils.get_all_children(model)[:-1]:
            if self._FEATURE_HOOK.needs_propagation(child):
                self._FEATURE_HOOK.propagate(child, inputs[0])

            if targets not in (None, {}):
                if self._TARGET_HOOK.needs_propagation(child):
                    self._TARGET_HOOK.propagate(child, targets)

        return inputs, {}

    def post_forward(self, model, outputs):
        """Forward pass for the DataPropagationHook.

        This function is called when the hook is executed
        during the forward pass of the parent module.

        Args:
            model (nn.Module): The parent module for which the hook is registered.
            inputs: The inputs to the parent module.
            kwargs: Additional keyword arguments to the parent module.
                This is used to retrieve the targets.

        Returns:
            Tuple: The original inputs and empty dict for kwargs.
        """
        for child in module_utils.get_all_children(model)[:-1]:
            if self._FEATURE_HOOK.needs_propagation(child):
                for key in _get_features(child, rename=False).keys():
                    delattr(child, key)

            if self._TARGET_HOOK.needs_propagation(child):
                targets = _get_targets(child, rename=False)
                if isinstance(targets, dict):
                    for key in _get_targets(child, rename=False).keys():
                        if hasattr(child, key):
                            delattr(child, key)
                else:
                    if hasattr(child, _TARGET_PREFIX):
                        delattr(child, _TARGET_PREFIX)

        return outputs


def needs_data_propagation_hook(model: nn.Module) -> bool:
    for child in module_utils.get_all_children(model):
        if hasattr(child, _FeatureReceiverHook.PROPERTY_NAME):
            return True
        if hasattr(child, _TargetReceiverHook.PROPERTY_NAME):
            return True

    return False


def register_data_propagation_hook(model: nn.Module) -> _DataPropagationHook:
    """Register a data propagation hook for a PyTorch module.

    Args:
        model (nn.Module): The model to register the data propagation hook for.

    Returns:
        DataPropagationHook: The registered data propagation hook.
    """
    hook = _DataPropagationHook()

    model.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    model.register_forward_hook(hook, with_kwargs=True)

    return hook


def register_feature_hook(module: nn.Module, schema: Schema):
    hook = _FeatureReceiverHook(schema)

    module.register_forward_pre_hook(hook, with_kwargs=True)
    module.register_buffer(hook.PROPERTY_NAME, torch.tensor(True), persistent=False)
    _input_schema = getattr(module, "input_schema", None)
    module.input_schema = _input_schema + schema if _input_schema else schema

    return hook


def register_target_hook(module: nn.Module):
    hook = _TargetReceiverHook()

    module.register_forward_pre_hook(hook, with_kwargs=True)
    module.register_buffer(hook.PROPERTY_NAME, torch.tensor(True), persistent=False)

    return hook


def _get_features(self: nn.Module, rename=True) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """Retrieve the features from the buffers of a PyTorch module.

    Args:
        module (nn.Module): The module containing the features.

    Returns:
        Union[Dict[str, torch.Tensor], torch.Tensor]:
            The features from the module, either as a dictionary of
            named tensors or a single tensor.
    """

    prefix = _FEATURE_PREFIX
    features = {}

    for name, buffer in self.named_buffers():
        if name.startswith(prefix):
            features[name] = buffer

    if not features:
        raise RuntimeError(
            "No feature buffers found. Ensure that `register_data_propagation_hook` has been "
            "called on the parent module with `propagate_features=True`. For example:\n\n"
            "    register_data_propagation_hook(model, propagate_features=True)"
        )

    if len(features) == 1:
        return list(features.values())[0]

    if not rename:
        return features

    return {k[len(prefix) + 1 :]: v for k, v in features.items()}


def _get_targets(
    self: nn.Module, strict=False, rename=True
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """Retrieve the targets from the buffers of a PyTorch module.

    Args:
        module (nn.Module): The module containing the targets.

    Returns:
        Union[Dict[str, torch.Tensor], torch.Tensor]:
            The targets from the module, either as a dictionary of
            named tensors or a single tensor.
    """

    prefix = _TARGET_PREFIX
    targets = {}

    for name, buffer in self.named_buffers():
        if name.startswith(prefix):
            targets[name] = buffer

    if not targets and not strict:
        return None

    if not targets and strict:
        raise RuntimeError(
            "No targets buffers found. Ensure that `register_data_propagation_hook` has been "
            "called on the parent module with `propagate_targets=True`. For example:\n\n"
            "    register_data_propagation_hook(model, propagate_targets=True)"
        )

    if len(targets) == 1 and hasattr(self, prefix):
        return list(targets.values())[0]

    if not rename:
        return targets

    return {k[len(prefix) + 1 :]: v for k, v in targets.items()}


def _upsert_buffer(module, name, data, persistent=False):
    if hasattr(module, name):
        setattr(module, name, data)
    else:
        module.register_buffer(name, data, persistent=persistent)


def _upsert_buffers(
    child: nn.Module, data: Union[Dict[str, torch.Tensor], torch.Tensor], prefix: str
):
    """Update or insert buffers for the given child module.

    Args:
        child (nn.Module): The child module to update or insert buffers for.
        data (Union[Dict[str, torch.Tensor], torch.Tensor]): The data to be added as buffers.
        prefix (str): The prefix to be used for naming the buffers.
    """
    if isinstance(child, nn.ModuleList):
        for c in child:
            _upsert_buffers(c, data, prefix)
    elif isinstance(data, dict):
        for key, val in data.items():
            key_prefix = f"{prefix}_{key}"
            _upsert_buffers(child, val, key_prefix)
    else:
        name = f"__buffer_{prefix}"
        _upsert_buffer(child, name, data, persistent=True)


# Cached
@lru_cache(maxsize=1000)
def _count_function_params(func):
    # Get the signature of the function
    func_signature = inspect.signature(func)

    # Initialize counters for args and kwargs
    args_count = 0
    kwargs_count = 0
    has_args = False
    has_kwargs = False

    # Iterate through the parameters of the function
    for param in func_signature.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if param.default == inspect.Parameter.empty:
                args_count += 1
            else:
                kwargs_count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_args = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True

    return args_count, kwargs_count, has_args, has_kwargs
