import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from merlin.schema import Schema


def ToTuple(schema: Schema) -> "ToTupleModule":
    """
    Creates a ToTupleModule for a given schema.

    This function is especially useful for serving models with Triton,
    as Triton doesn't allow models that output a dictionary. Instead,
    by using this function, models can be modified to output tuples.

    Parameters
    ----------
    schema : Schema
        Input schema for which a ToTupleModule is to be created.

    Returns
    -------
    ToTupleModule
        A ToTupleModule corresponding to the length of the given schema.
        The output can vary from ToTuple1 to ToTuple10.

    Raises
    ------
    ValueError
        If the length of the schema is more than 10,
        a ValueError is raised with an appropriate error message.

    Example usage ::
    >>> import torch
    >>> schema = Schema(["a", "b", "c"])
    >>> ToTupleModule = ToTuple(schema)
    >>> tensor_dict = {'a': torch.tensor([1]), 'b': torch.tensor([2.]), 'c': torch.tensor([2.])}
    >>> output = ToTupleModule(tensor_dict)
    >>> print(output)
    (tensor([1]), tensor([2.]), tensor([2.]))
    """
    schema_length = len(schema)

    if schema_length <= 10:
        ToTupleClass = getattr(sys.modules[__name__], f"ToTuple{schema_length}")
        return ToTupleClass(input_schema=schema)
    else:
        raise ValueError(f"Cannot convert schema of length {schema_length} to a tuple")


class ToTupleModule(nn.Module):
    def __init__(self, input_schema: Optional[Schema] = None):
        super().__init__()
        if input_schema is not None:
            self.initialize_from_schema(input_schema)
            self._initialized_from_schema = True

    def initialize_from_schema(self, input_schema: Schema):
        self._input_schema = input_schema
        self._column_names = input_schema.column_names

    def value_list(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []

        if not hasattr(self, "_column_names"):
            raise RuntimeError("initialize_from_schema() must be called before value_list()")

        for col in self._column_names:
            outputs.append(inputs[col])

        return outputs


class ToTuple1(ToTupleModule):
    """Converts a dictionary of tensors of length=1 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0],)


class ToTuple2(ToTupleModule):
    """Converts a dictionary of tensors of length=2 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1])


class ToTuple3(ToTupleModule):
    """Converts a dictionary of tensors of length=3 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2])


class ToTuple4(ToTupleModule):
    """Converts a dictionary of tensors of length=4 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3])


class ToTuple5(ToTupleModule):
    """Converts a dictionary of tensors of length=5 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4])


class ToTuple6(ToTupleModule):
    """Converts a dictionary of tensors of length=6 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5])


class ToTuple7(ToTupleModule):
    """Converts a dictionary of tensors of length=7 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5], _list[6])


class ToTuple8(ToTupleModule):
    """Converts a dictionary of tensors of length=8 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5], _list[6], _list[7])


class ToTuple9(ToTupleModule):
    """Converts a dictionary of tensors of length=9 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _list = list(inputs.values())
        return (
            _list[0],
            _list[1],
            _list[2],
            _list[3],
            _list[4],
            _list[5],
            _list[6],
            _list[7],
            _list[8],
        )


class ToTuple10(ToTupleModule):
    """Converts a dictionary of tensors of length=10 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[
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
    ]:
        _list = list(inputs.values())
        return (
            _list[0],
            _list[1],
            _list[2],
            _list[3],
            _list[4],
            _list[5],
            _list[6],
            _list[7],
            _list[8],
            _list[9],
        )
