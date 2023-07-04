from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from merlin.schema import Schema


class _ToTuple(nn.Module):
    def __init__(self, input_schema: Optional[Schema] = None):
        super().__init__()
        if input_schema is not None:
            self.setup_schema(input_schema)

    def setup_schema(self, input_schema: Schema):
        self._input_schema = input_schema
        self._column_names = input_schema.column_names

    def value_list(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []

        if not hasattr(self, "_column_names"):
            raise RuntimeError("setup_schema() must be called before value_list()")

        for col in self._column_names:
            outputs.append(inputs[col])

        return outputs


class ToTuple1(_ToTuple):
    """Converts a dictionary of tensors of length=1 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0],)


class ToTuple2(_ToTuple):
    """Converts a dictionary of tensors of length=2 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1])


class ToTuple3(_ToTuple):
    """Converts a dictionary of tensors of length=3 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2])


class ToTuple4(_ToTuple):
    """Converts a dictionary of tensors of length=4 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3])


class ToTuple5(_ToTuple):
    """Converts a dictionary of tensors of length=5 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4])


class ToTuple6(_ToTuple):
    """Converts a dictionary of tensors of length=6 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = self.value_list(inputs)
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5])


class ToTuple7(_ToTuple):
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


class ToTuple8(_ToTuple):
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


class ToTuple9(_ToTuple):
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


class ToTuple10(_ToTuple):
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
