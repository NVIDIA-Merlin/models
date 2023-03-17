from typing import Dict, Union

import torch
from torch import nn


class ConcatFeatures(nn.Module):
    """Concatenate tensors along a specified dimension.

    Attributes
    ----------
    dim : int
        The dimension along which the tensors will be concatenated.

    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenates input tensors along the specified dimension.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary where keys are the names of the tensors
            and values are the tensors to be concatenated.

        Returns
        -------
        torch.Tensor
            A tensor that is the result of concatenating
            the input tensors along the specified dimension.

        Raises
        ------
        RuntimeError
            If the input tensor shapes don't match for concatenation
            along the specified dimension.
        """
        sorted_tensors = [inputs[name] for name in sorted(inputs.keys())]
        if not all(
            (
                t.shape[: self.dim] == sorted_tensors[0].shape[: self.dim]
                and t.shape[self.dim + 1 :] == sorted_tensors[0].shape[self.dim + 1 :]
            )
            for t in sorted_tensors
        ):
            raise RuntimeError(
                "Input tensor shapes don't match for concatenation",
                "along the specified dimension.",
            )

        return torch.cat(sorted_tensors, dim=self.dim)


class SumResidual(nn.Module):
    """
    SumResidual module applies a residual connection to input tensors, where the skip-connection is
    part of the "shortcut" key in the input dictionary. Residual connections help mitigate the
    vanishing gradient problem and allow for training of deeper networks.

    Parameters
    ----------
    activation : str or callable, optional, default="relu"
        The activation function to apply after the residual connection. If it is a string, it should
        be the name of an activation function available in the `torch` library. If it is a callable
        object, it should be a function that takes a tensor and returns a tensor.
    shortcut_name : str, optional, default="shortcut"
        The key in the input dictionary that corresponds to the skip-connection tensor.

    Attributes
    ----------
    activation : callable
        The activation function to apply after the residual connection.
    shortcut_name : str
        The key in the input dictionary that corresponds to the skip-connection tensor.
    """

    def __init__(self, activation="relu", shortcut_name="shortcut"):
        super().__init__()

        # Check if activation is a string or a callable object
        if activation and not isinstance(activation, str) and not callable(activation):
            raise ValueError("activation should be a string or a callable object")

        if isinstance(activation, str):
            if not hasattr(torch, activation):
                raise ValueError(
                    f"torch does not have the specified activation function: {activation}"
                )
            activation = getattr(torch, activation)

        # Check if shortcut_name is a string
        if not isinstance(shortcut_name, str):
            raise ValueError("shortcut_name should be a string")

        self.activation = activation
        self.shortcut_name = shortcut_name

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies the residual connection and activation function to the input tensors.

        Parameters
        ----------
        inputs : dict of str: torch.Tensor
            A dictionary with keys as strings and values as tensors. The dictionary should contain
            a key matching the `shortcut_name` attribute, which corresponds to the skip-connection
            tensor.

        Returns
        -------
        torch.Tensor or dict of str: torch.Tensor
            If there is only one output tensor, it is returned directly. Otherwise, a dictionary of
            tensors is returned, where the keys match the input dictionary keys, except for the
            `shortcut_name` key.
        """
        if self.shortcut_name not in inputs:
            raise RuntimeError(
                f"Shortcut '{self.shortcut_name}' not found in the inputs dictionary"
            )

        shortcut = inputs[self.shortcut_name]
        outputs = {}
        for key, val in inputs.items():
            if key == self.shortcut_name:
                continue

            if val.shape != shortcut.shape:
                raise RuntimeError(
                    f"Shape mismatch: {key} tensor ({val.shape}) and shortcut ",
                    f"tensor ({shortcut.shape}) must have the same shape",
                )

            residual = val + shortcut
            if self.activation:
                residual = self.activation(residual)
            outputs[key] = residual

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs
