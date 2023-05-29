from typing import Dict, Union

import torch
from torch import nn

from merlin.models.torch.registry import registry


@registry.register("concat")
class Concat(nn.Module):
    """Concatenate tensors along a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension along which the tensors will be concatenated.
        Default is -1.

    Examples
    --------
    >>> concat = Concat()
    >>> feature1 = torch.tensor([[1, 2], [3, 4]])  # Shape: [batch_size, feature_dim]
    >>> feature2 = torch.tensor([[5, 6], [7, 8]])  # Shape: [batch_size, feature_dim]
    >>> input_dict = {"feature1": feature1, "feature2": feature2}
    >>> output = concat(input_dict)
    >>> print(output)
    tensor([[1, 2, 5, 6],
             [3, 4, 7, 8]])  # Shape: [batch_size, feature_dim*number_of_features]

    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenates input tensors along the specified dimension.

        The input dictionary will be sorted by name before concatenation.

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
        # TODO: Fix this for dim=-1
        if self.dim > 0:
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


@registry.register("stack")
class Stack(nn.Module):
    """Stack tensors along a specified dimension.

    The input dictionary will be sorted by name before concatenation.

    Parameters
    ----------
    dim : int
        The dimension along which the tensors will be stacked.
        Default is 0.

    Examples
    --------
    >>> stack = Stack()
    >>> feature1 = torch.tensor([[1, 2], [3, 4]])  # Shape: [batch_size, feature_dim]
    >>> feature2 = torch.tensor([[5, 6], [7, 8]])  # Shape: [batch_size, feature_dim]
    >>> input_dict = {"feature1": feature1, "feature2": feature2}
    >>> output = stack(input_dict)
    >>> print(output)
    tensor([[[1, 2],
             [5, 6]],

            [[3, 4],
             [7, 8]]])  # Shape: [batch_size, number_of_features, feature_dim]


    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Stacks input tensors along the specified dimension.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary where keys are the names of the tensors
            and values are the tensors to be stacked.

        Returns
        -------
        torch.Tensor
            A tensor that is the result of stacking
            the input tensors along the specified dimension.

        Raises
        ------
        RuntimeError
            If the input tensor shapes don't match for stacking.
        """
        sorted_tensors = [inputs[name] for name in sorted(inputs.keys())]
        if not all(t.shape == sorted_tensors[0].shape for t in sorted_tensors):
            raise RuntimeError("Input tensor shapes don't match for stacking.")

        return torch.stack(sorted_tensors, dim=self.dim)


class MaybeAgg(nn.Module):
    """
    This class is designed to conditionally apply an aggregation operation
    (e.g., Stack or Concat) on a tensor or a dictionary of tensors.

    Parameters
    ----------
    agg : nn.Module
        The aggregation operation to be applied.

    Examples
    --------
    >>> stack = Stack(dim=0)
    >>> maybe_agg = MaybeAgg(agg=stack)
    >>> tensor1 = torch.tensor([[1, 2], [3, 4]])
    >>> tensor2 = torch.tensor([[5, 6], [7, 8]])
    >>> input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    >>> output = maybe_agg(input_dict)
    >>> print(output)
    tensor([[[1, 2],
             [3, 4]],

            [[5, 6],
             [7, 8]]])

    >>> tensor = torch.tensor([1, 2, 3])
    >>> output = maybe_agg(tensor)
    >>> print(output)
    tensor([1, 2, 3])
    """

    def __init__(self, agg: nn.Module):
        super().__init__()
        self.agg = agg

    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Conditionally applies the aggregation operation on the inputs.

        Parameters
        ----------
        inputs : Union[Dict[str, torch.Tensor], torch.Tensor]
            Inputs to be aggregated. If inputs is a dictionary of tensors,
            the aggregation operation will be applied. If inputs is a single tensor,
            it will be returned as is.

        Returns
        -------
        torch.Tensor
            Aggregated tensor if inputs is a dictionary, otherwise the input tensor.
        """

        if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            return self.agg(inputs)

        if not torch.jit.isinstance(inputs, torch.Tensor):
            raise RuntimeError("Inputs must be either a dictionary of tensors or a single tensor.")

        return inputs
