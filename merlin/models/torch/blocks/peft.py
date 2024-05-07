# flake8: noqa

import math
from typing import Final, TypeVar

import torch
from torch import nn

from merlin.models.torch.block import Block

BlockT = TypeVar("BlockT", bound=Block)


class LoRA(nn.Module):
    """Low Ranking Adaptation for parameter-efficient fine-tuning.

    Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects
    trainable rank decomposition matrices into each layer of the Transformer architecture,
    greatly reducing the number of trainable parameters for downstream tasks.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘


    Example usage::
        block = Block(...).freeze()
        block_with_lora = LoRA.walk(block, r=8, lora_alpha=0.5, lora_dropout=0.1)


    Parameters
    __________
    module : nn.Module
        The module to apply LoRA to.
    r : int, optional
        The rank of the LoRA approximation. Default is 0.
    lora_alpha : float, optional
        The scaling factor for the LoRA approximation. Default is 1.
    lora_dropout : float, optional
        The dropout rate for the LoRA approximation. Default is 0.
    merge_weights : bool, optional
        Whether to merge the LoRA weights into the module weights. Default is True.
    """

    wrapped_name: Final[str]

    def __init__(
        self,
        module: nn.Module,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        super().__init__()
        self.module = module
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merge_weights = merge_weights
        self.merged = False
        self.wrapped_name = self.module.__class__.__name__

        if self.wrapped_name == "Linear":
            self.in_features = self.module.in_features
            self.out_features = self.module.out_features
        elif self.wrapped_name == "Embedding":
            self.in_features = self.module.num_embeddings
            self.out_features = self.module.embedding_dim
        else:
            raise ValueError(f"LoRA is not supported for {type(self.module)} modules.")

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features))
            self.scaling = self.lora_alpha / self.r
            self.module.weight.requires_grad = False

    @staticmethod
    @torch.jit.ignore
    def walk(
        block: BlockT,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ) -> BlockT:
        """
        Applies the LoRA approximation to the given block.

        The LoRA instances are created by walking through the module tree of the given block
        and replacing every `nn.Linear` or `nn.Embedding` module encountered with a corresponding
        LoRA instance.

        The parameters are set based on the input arguments and used to create a low-rank
        approximation of the original weight matrix in the module. This approximation is intended
        to reduce the complexity of the model without significant loss of performance.

        Example usage::
            block = Block(...).freeze()
            block_with_lora = LoRA.apply(block, r=8, lora_alpha=0.5, lora_dropout=0.1)

        Parameters
        ----------
        block : BlockT
            The block to apply the LoRA to.
        r : int, optional
            The rank of the LoRA approximation. Default is 0.
        lora_alpha : float, optional
            The scaling factor for the LoRA approximation. Default is 1.
        lora_dropout : float, optional
            The dropout rate for the LoRA approximation. Default is 0.
        merge_weights : bool, optional
            Whether to merge the LoRA weights into the module weights.
            Default is True.

        Returns
        -------
        BlockT
            The block with the applied LoRA.
        """

        def to_apply(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                return LoRA(
                    module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=merge_weights,
                )

            return module

        return block.walk(to_apply)

    def reset_parameters(self):
        """
        Resets the parameters of the module and LoRA approximation.
        """
        self.module.reset_parameters()
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """
        Sets the module in training mode.

        This method overrides the base method in nn.Module to allow the addition and subtraction
        of the low rank approximation (LoRA) to/from the weights of the module, depending on the
        mode.

        In training mode (mode=True), if merge_weights is set to True and the weights have been
        merged,it subtracts the LoRA from the weights. This allows the original weights to be
        exposed during training.

        In evaluation mode (mode=False), if merge_weights is set to True and the weights haven't
        been merged, it adds the LoRA to the weights. This allows the LoRA to augment the weights
        during evaluation.

        Parameters
        ----------
        mode : bool, optional
            If True, sets the module in training mode, else sets it in evaluation mode.
            Default is True.
        """
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    if self.wrapped_name == "Linear":
                        self.module.weight.data -= (self.lora_A @ self.lora_B) * self.scaling
                    else:  # Embedding
                        self.module.weight.data -= (self.lora_A @ self.lora_B) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    if self.wrapped_name == "Linear":
                        self.module.weight.data += (self.lora_A @ self.lora_B).t() * self.scaling
                    else:  # Embedding
                        self.module.weight.data += (self.lora_A @ self.lora_B) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)
        if self.r > 0 and not self.merged:
            if self.wrapped_name == "Linear":
                W_lora = self.lora_A @ self.lora_B * self.scaling
                lora_output = x @ W_lora  # this computes the LoRA output
                output += self.lora_dropout(lora_output)  # add the dropout on LoRA output
            else:  # Embedding
                # For an embedding layer, we cannot use the input `x` to multiply `W_lora` directly.
                # Instead, we need to look up the LoRA output
                # from `W_lora` using `x` as the indices, and then
                # add the dropout on the LoRA output.
                W_lora = self.lora_A @ self.lora_B * self.scaling
                lora_output = W_lora[x, :]
                output += self.lora_dropout(lora_output)

        return output
