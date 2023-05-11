from typing import Dict, Final, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.utils import module_utils


class TorchScriptWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    accepts_dict: Final[bool]

    def __init__(self, to_wrap: nn.Module):
        super().__init__()

        if not isinstance(to_wrap, nn.Module):
            raise ValueError("Expected a nn.Module, but got something else instead.")

        self.to_wrap = to_wrap
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)
        self.accepts_dict = module_utils.is_tabular(to_wrap)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    # else-clause is needed to make torchscript happy
                    _batch = batch if batch is not None else Batch({})

                    if not self.accepts_dict:
                        if not torch.jit.isinstance(inputs, torch.Tensor):
                            raise RuntimeError("Expected a tensor, but got a dictionary instead.")
                        x: torch.Tensor = (
                            inputs if isinstance(inputs, torch.Tensor) else inputs["x"]
                        )

                        return self.to_wrap(x, batch=_batch)

                    else:
                        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                            raise RuntimeError("Expected a dictionary, but got a tensor instead.")
                        x: Dict[str, torch.Tensor] = inputs

                        return self.to_wrap(x, batch=_batch)

        if not self.accepts_dict:
            if not torch.jit.isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["x"]

            return self.to_wrap(x)

        else:
            if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
                raise RuntimeError("Expected a dictionary, but got a tensor instead.")
            x: Dict[str, torch.Tensor] = inputs

            return self.to_wrap(x)

    def unwrap(self) -> nn.Module:
        return self.to_wrap

    def __getattr__(self, name: str):
        if name in [
            "accepts_dict",
            "accepts_batch",
            "requires_batch",
            "to_wrap",
            "unwrap",
        ]:
            return super().__getattr__(name)

        # Otherwise, return the attribute from the to_wrap module
        return getattr(self.to_wrap, name)

    def __repr__(self):
        return self.to_wrap.__repr__()
