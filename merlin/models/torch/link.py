from typing import Dict, Union

import torch
from torch import nn

from merlin.models.torch.registry import registry


class Link(nn.Module):
    @classmethod
    def parse(cls, input: Union[str, "Link"]) -> "Link":
        if isinstance(input, str):
            input = registry.get(input)
        if not isinstance(input, Link):
            raise RuntimeError(f"Expected a Link, but got {type(input)} instead.")

        return input

    def setup_link(self, input: nn.Module, output: nn.Module) -> "Link":
        self.input = input
        self.output = output

        return self


@registry.register("residual")
class Residual(Link):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.output(x)


@registry.register("shortcut")
class Shortcut(Link):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.output(x), "shortcut": x}


@registry.register("shortcut-concat")
class ShortcutConcat(Link):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.output(x)
        return torch.cat((x, intermediate_output), dim=1)
