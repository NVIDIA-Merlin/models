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

from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch import schema
from merlin.models.torch.batch import Batch
from merlin.models.torch.container import BlockContainer
from merlin.models.torch.registry import registry


class AggModule(nn.Module, schema.Selectable):
    def select(self, selection: schema.Selection) -> "AggModule":
        if not hasattr(self, "schema"):
            raise ValueError(f"Schema not set in {self}, so cannot select.")

        selected = schema.select(self.schema, selection)
        if selected == self.schema:
            return self

        diff = set(selected.column_names) - set(self.schema.column_names)

        raise ValueError(f"Sub-selecting {diff} from {self} is not supported. ")

    def extra_repr(self) -> str:
        if getattr(self, "schema", None):
            return f"{', '.join(self.schema.column_names)}"

        return ""


@registry.register("concat")
class Concat(AggModule):
    """Concatenate tensors along a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension along which the tensors will be concatenated.
        Default is -1.
    align_dims: bool, default = True
        If True, adds an extra dimension to all input tensors that need it.


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

    def __init__(self, dim: int = -1, align_dims: bool = True):
        super().__init__()
        self.dim = dim
        self.align_dims = align_dims

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

        if self.align_dims:
            max_dims = max(tensor.dim() for tensor in sorted_tensors)
            max_dims = max(
                max_dims, 2
            )  # assume first dimension is batch size + at least one feature
            _sorted_tensors = []
            for tensor in sorted_tensors:
                if tensor.dim() < max_dims:
                    _sorted_tensors.append(tensor.unsqueeze(-1))
                else:
                    _sorted_tensors.append(tensor)
            sorted_tensors = _sorted_tensors

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

        return torch.cat(sorted_tensors, dim=self.dim).float()


@registry.register("stack")
class Stack(AggModule):
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

        return torch.stack(sorted_tensors, dim=self.dim).float()


@registry.register("element-wise-sum")
class ElementWiseSum(AggModule):
    """Element-wise sum of tensors.

    The input dictionary will be sorted by name before concatenation.
    The sum is computed along the first dimension (default for Stack class).

    Example usage::
        >>> ewsum = ElementWiseSum()
        >>> feature1 = torch.tensor([[1, 2], [3, 4]])  # Shape: [batch_size, feature_dim]
        >>> feature2 = torch.tensor([[5, 6], [7, 8]])  # Shape: [batch_size, feature_dim]
        >>> input_dict = {"feature1": feature1, "feature2": feature2}
        >>> output = ewsum(input_dict)
        >>> print(output)
        tensor([[ 6,  8],
                [10, 12]])  # Shape: [batch_size, feature_dim]

    """

    def __init__(self):
        super().__init__()
        self.stack = Stack(dim=0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs an element-wise sum of input tensors.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary where keys are the names of the tensors
            and values are the tensors to be summed.

        Returns
        -------
        torch.Tensor
            A tensor that is the result of performing an element-wise sum
            of the input tensors.

        Raises
        ------
        RuntimeError
            If the input tensor shapes don't match for stacking.
        """
        return self.stack(inputs).sum(dim=0)


class MaybeAgg(BlockContainer):
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
        super().__init__(agg)

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        batch: Optional[Batch] = None,
    ) -> torch.Tensor:
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
            return self.values[0](inputs)

        if not torch.jit.isinstance(inputs, torch.Tensor):
            raise RuntimeError("Inputs must be either a dictionary of tensors or a single tensor.")

        return inputs
