from copy import deepcopy
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block


class CrossAttentionBlock(Block):
    """
    Cross Attention Block module which performs a multihead attention operation
    on a provided context and sequence.

    Note this block assumes that the input and output tensors are provided as
    (batch, seq, feature). When using modules provided in PyTorch, e.g.,
    ``torch.nn.MultiheadAttention``, the ``batch_first`` parameter should be
    set to True to match the shape.

    Example usage
    -------------

    >>> cross = CrossAttentionBlock(
    ...    attention=nn.MultiheadAttention(10, 2, batch_first=True),
    ...    key="context",
    ...    seq_key="sequence",
    ... )
    >>> input_dict = {
    ...     "context": torch.randn(1, 2, 10),
    ...     "sequence": torch.randn(1, 6, 10)}
    ... }
    >>> cross(input_dict)

    Parameters
    ----------
    module : nn.Module
        Variable length input module list.
    attention : nn.MultiheadAttention, optional
        Predefined multihead attention module. If not provided, it's inferred from the first module.
    name : str, optional
        Name for the block.
    key : str, optional
        Key for the context tensor in the input dictionary.
    seq_key : str, optional
        Key for the sequence tensor in the input dictionary.
    """

    def __init__(
        self,
        *module: nn.Module,
        attention: Optional[nn.MultiheadAttention] = None,
        name: str = None,
        key: str = "context",
        seq_key: Optional[str] = None,
    ):
        super().__init__(*module, name=name)

        self.key = key
        self.seq_key = seq_key
        if attention is None:
            if not (
                hasattr(module[0], "d_model")
                and hasattr(module[0], "nhead")
                and hasattr(module[0], "dropout")
            ):
                raise ValueError("Attention module not provided and cannot be inferred from module")

            # Try to infer from module
            cross_attention = nn.MultiheadAttention(
                module[0].d_model, module[0].nhead, module[0].dropout
            )
        else:
            cross_attention = attention

        self.cross_attention = nn.ModuleList([cross_attention])
        if len(module) > 1:
            for m in module:
                self.cross_attention.append(
                    m.copy() if hasattr(m, "copy") else deepcopy(cross_attention)
                )

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        """
        Perform forward pass of the CrossAttentionBlock.

        Parameters
        ----------
        inputs : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Dictionary containing the input tensors.
        batch : Optional[Batch]
            Optional batch information for the forward pass.

        Returns
        -------
        torch.Tensor
            Output tensor after the multihead attention operation.

        Raises
        ------
        ValueError
            If the input is a torch.Tensor instead of a dictionary.
        """

        if isinstance(inputs, torch.Tensor):
            raise ValueError("CrossAttentionBlock requires a dictionary input")

        context, sequence = self.get_context(inputs), self.get_seq(inputs)

        for module, attention in zip(self.values, self.cross_attention):
            sequence, _ = attention(sequence, context, context)
            sequence = module(sequence, batch=batch)

        return sequence

    def get_context(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Retrieve the context tensor from the input dictionary using the key.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Input dictionary containing the tensors.

        Returns
        -------
        torch.Tensor
            The context tensor.
        """
        return x[self.key]

    def get_seq(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Retrieve the sequence tensor from the input dictionary using the key.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Input dictionary containing the tensors.

        Returns
        -------
        torch.Tensor
            The sequence tensor.

        Raises
        ------
        RuntimeError
            If the seq_key is not found in the input dictionary or if the dictionary has more
            than 2 keys and seq_key is not defined.
        """
        if self.seq_key is None:
            if len(x) == 2:
                for key in x.keys():
                    if key != self.key:
                        return x[key]
            else:
                raise RuntimeError(
                    "Please set seq_key for when more than 2 keys are present ",
                    f"in the input dictionary, got: {x}.",
                )

        if self.seq_key not in x:
            raise RuntimeError(f"Could not find {self.seq_key} in input dictionary, got: {x}.")

        return x[self.seq_key]
