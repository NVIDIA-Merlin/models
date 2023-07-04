from typing import Dict, Tuple

import torch
from torch import nn


class ToTuple1(nn.Module):
    """Converts a dictionary of tensors of length=1 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0],)


class ToTuple2(nn.Module):
    """Converts a dictionary of tensors of length=2 to a tuple of tensors."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0], _list[1])


class ToTuple3(nn.Module):
    """Converts a dictionary of tensors of length=3 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2])


class ToTuple4(nn.Module):
    """Converts a dictionary of tensors of length=4 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2], _list[3])


class ToTuple5(nn.Module):
    """Converts a dictionary of tensors of length=5 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2], _list[3], _list[4])


class ToTuple6(nn.Module):
    """Converts a dictionary of tensors of length=6 to a tuple of tensors."""

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5])


class ToTuple7(nn.Module):
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
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5], _list[6])


class ToTuple8(nn.Module):
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
        _list = list(inputs.values())
        return (_list[0], _list[1], _list[2], _list[3], _list[4], _list[5], _list[6], _list[7])


class ToTuple9(nn.Module):
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


class ToTuple10(nn.Module):
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
