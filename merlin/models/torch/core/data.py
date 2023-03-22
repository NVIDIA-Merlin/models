from typing import Dict, Union

import torch
from torch import nn

from merlin.models.torch.utils import module_utils


class DataPropagationHook(nn.Module):
    """A data propagation hook for PyTorch modules.

    This hook allows you to propagate features and/or targets through
    the children of a model during the forward pass.

    Args:
        propagate_features (bool, optional): Whether to propagate features.
            Defaults to True.
        propagate_targets (bool, optional): Whether to propagate targets.
            Defaults to True.
    """

    def __init__(self, propagate_features: bool = True, propagate_targets: bool = True):
        super().__init__()
        self.propagate_features = propagate_features
        self.propagate_targets = propagate_targets

    def forward(self, model, inputs, kwargs):
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
            if self.propagate_features:
                self._upsert_buffers(child, inputs[0], "feature")
            if targets not in (None, {}) and self.propagate_targets:
                self._upsert_buffers(child, targets, "target")

        return inputs, {}

    def _upsert_buffers(
        self, child: nn.Module, data: Union[Dict[str, torch.Tensor], torch.Tensor], prefix: str
    ):
        """Update or insert buffers for the given child module.

        Args:
            child (nn.Module): The child module to update or insert buffers for.
            data (Union[Dict[str, torch.Tensor], torch.Tensor]): The data to be added as buffers.
            prefix (str): The prefix to be used for naming the buffers.
        """
        if isinstance(child, nn.ModuleList):
            for c in child:
                self._upsert_buffers(c, data, prefix)
        elif isinstance(data, dict):
            for key, val in data.items():
                key_prefix = f"{prefix}_{key}"
                self._upsert_buffers(child, val, key_prefix)
        else:
            name = f"__buffer_{prefix}"
            if hasattr(child, name):
                setattr(child, name, data)
            else:
                child.register_buffer(name, data, persistent=False)


def register_data_propagation_hook(
    model: nn.Module,
    propagate_features: bool = False,
    propagate_targets: bool = False,
) -> DataPropagationHook:
    """Register a data propagation hook for a PyTorch module.

    Args:
        model (nn.Module): The model to register the data propagation hook for.
        propagate_features (bool, optional): Whether to propagate features. Defaults to False.
        propagate_targets (bool, optional): Whether to propagate targets. Defaults to False.

    Returns:
        DataPropagationHook: The registered data propagation hook.
    """
    hook = DataPropagationHook(propagate_features, propagate_targets)

    model.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)

    return hook


def get_features(module: nn.Module) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """Retrieve the features from the buffers of a PyTorch module.

    Args:
        module (nn.Module): The module containing the features.

    Returns:
        Union[Dict[str, torch.Tensor], torch.Tensor]:
            The features from the module, either as a dictionary of
            named tensors or a single tensor.
    """

    prefix = "__buffer_feature"
    features = {}

    for name, buffer in module.named_buffers():
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

    return {k[len(prefix) + 1 :]: v for k, v in features.items()}


def get_targets(module: nn.Module) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """Retrieve the targets from the buffers of a PyTorch module.

    Args:
        module (nn.Module): The module containing the targets.

    Returns:
        Union[Dict[str, torch.Tensor], torch.Tensor]:
            The targets from the module, either as a dictionary of
            named tensors or a single tensor.
    """

    prefix = "__buffer_target"
    targets = {}

    for name, buffer in module.named_buffers():
        if name.startswith(prefix):
            targets[name] = buffer

    if not targets:
        raise RuntimeError(
            "No targets buffers found. Ensure that `register_data_propagation_hook` has been "
            "called on the parent module with `propagate_targets=True`. For example:\n\n"
            "    register_data_propagation_hook(model, propagate_targets=True)"
        )

    if len(targets) == 1:
        return list(targets.values)[0]

    return {k[len(prefix) + 1 :]: v for k, v in targets.items()}
