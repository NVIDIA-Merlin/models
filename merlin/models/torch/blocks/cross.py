from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from merlin.models.torch.block import Block
from merlin.models.torch.transforms.agg import Concat
from merlin.models.utils.doc_utils import docstring_parameter

_DCNV2_REF = """
    References
    ----------
    .. [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf

"""


class LazyMirrorLinear(LazyModuleMixin, nn.Linear):
    """A :class:`torch.nn.Linear` module where both
    `in_features` & `out_features` are inferred. (i.e. `out_features` = `in_features`)

    Parameters
    ----------
    bias:
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``

    """

    cls_to_become = nn.Linear  # type: ignore[assignment]
    weight: nn.parameter.UninitializedParameter
    bias: nn.parameter.UninitializedParameter  # type: ignore[assignment]

    def __init__(self, bias: bool = True, device=None, dtype=None) -> None:
        # This code is taken from torch.nn.LazyLinear.__init__
        factory_kwargs = {"device": device, "dtype": dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False)
        self.weight = nn.parameter.UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = nn.parameter.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.out_features = self.in_features
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


@docstring_parameter(dcn_reference=_DCNV2_REF)
class CrossBlock(Block):
    """
    This block provides a way to create high-order feature interactions
    by a number of stacked Cross Layers, from DCN V2: Improved Deep & Cross Network [1].
    See Eq. (1) for full-rank and Eq. (2) for low-rank version.

    Parameters
    ----------
    depth : int, optional
        Number of cross-layers to be stacked, by default 1

    {dcn_reference}
    """

    def __init__(self, *module, name: Optional[str] = None):
        super().__init__(*module, name=name)
        self.concat = Concat()

    @classmethod
    def with_depth(cls, depth: int):
        if not depth > 0:
            raise ValueError(f"`depth` must be greater than 0, got {depth}")

        return cls(*Block(LazyMirrorLinear()).repeat(depth))

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            x = self.concat(inputs)
        else:
            x = inputs

        x0 = x
        current = x
        for module in self.values:
            module_out = module(current)
            if not isinstance(module_out, torch.Tensor):
                raise RuntimeError("CrossBlock expects a Tensor as output")

            current = x0 * module_out + current

        return current
