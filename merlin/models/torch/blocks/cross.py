from copy import deepcopy
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from merlin.models.torch.batch import Batch
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
                if not hasattr(self, "out_features") or self.out_features == 0:
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
    *module : nn.Module
        Variable length argument list of PyTorch modules to be contained in the block.
    name : Optional[str], default = None
        The name of the block. If None, no name is assigned.

    {dcn_reference}
    """

    def __init__(self, *module, name: Optional[str] = None):
        super().__init__(*module, name=name)
        self.concat = Concat()
        self.init_hook_handle = self.register_forward_pre_hook(self.initialize)

    @classmethod
    def with_depth(cls, depth: int) -> "CrossBlock":
        """Creates a CrossBlock with a given depth.

        Parameters
        ----------
        depth : int
            Depth of the CrossBlock.

        Returns
        -------
        CrossBlock
            A CrossBlock of the given depth.

        Raises
        ------
        ValueError
            If depth is less than or equal to 0.
        """
        if not depth > 0:
            raise ValueError(f"`depth` must be greater than 0, got {depth}")

        return cls(*Block(LazyMirrorLinear()).repeat(depth))

    @classmethod
    def with_low_rank(cls, depth: int, low_rank: nn.Module) -> "CrossBlock":
        """
        Creates a CrossBlock with a given depth and low rank. See Eq. (2) in [1].

        Parameters
        ----------
        depth : int
            Depth of the CrossBlock.
        low_rank : nn.Module
            Low rank module to include in the CrossBlock.

        Returns
        -------
        CrossBlock
            A CrossBlock of the given depth and low rank.
        """

        return cls(*(Block(deepcopy(low_rank), *block) for block in cls.with_depth(depth)))

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        """Forward-pass of the cross-block.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input data. It could be either a tensor or a dictionary of tensors.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.

        Raises
        ------
        RuntimeError
            If the output from a module is not a Tensor.
        """

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

    def initialize(self, module, inputs):
        """
        Initialize the block by setting the output features of all LazyMirrorLinear children.

        This is meant to be used as a forward pre-hook.

        Parameters
        ----------
        module : nn.Module
            The module to initialize.
        inputs : tuple
            The inputs to the forward method.
        """

        if torch.jit.isinstance(inputs[0], Dict[str, torch.Tensor]):
            _inputs = self.concat(inputs[0])
        else:
            _inputs = inputs[0]

        def set_out_features_lazy_mirror_linear(m):
            if isinstance(m, LazyMirrorLinear):
                m.out_features = _inputs.shape[-1]

        self.apply(set_out_features_lazy_mirror_linear)
        self.init_hook_handle.remove()  # Clear hook once block is initialized
