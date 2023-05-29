import copy
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.registry import TorchRegistryMixin

LinkType = Union[str, "Link"]


class Link(nn.Module, TorchRegistryMixin):
    """Base class for different types of network links.

    This is typically used as part of a `Block` to connect different modules.

    Some examples of links are:
        - `residual`: Adds the input to the output of the module.
        - `shortcut`: Outputs a dictionary with the output of the module and the input.
        - `shortcut-concat`: Concatenates the input and the output of the module.

    """

    def __init__(self, output: Optional[nn.Module] = None):
        super().__init__()

        if output is not None:
            self.setup_link(output)

    def setup_link(self, output: nn.Module) -> "Link":
        """
        Setup function for the link.

        Parameters
        ----------
        output : nn.Module
            The output module for the link.

        Returns
        -------
        Link
            The updated Link instance.
        """

        self.output = output

        return self

    def copy(self) -> "Link":
        """
        Returns a copy of the link.

        Returns
        -------
        Link
            The copied link.
        """
        return copy.deepcopy(self)


@Link.registry.register("residual")
class Residual(Link):
    """Adds the input to the output of the module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.output(x)


@Link.registry.register("shortcut")
class Shortcut(Link):
    """Outputs a dictionary with the output of the module and the input."""

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.output(x), "shortcut": x}


@Link.registry.register("shortcut-concat")
class ShortcutConcat(Link):
    """Concatenates the input and the output of the module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.output(x)
        return torch.cat((x, intermediate_output), dim=1)
