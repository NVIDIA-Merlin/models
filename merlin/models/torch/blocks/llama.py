from dataclasses import dataclass
from typing import Dict, Optional, TypeVar, Union

import torch
import torch.nn as nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.torch.blocks.attention import (
    AttentionMask,
    CausalSelfAttention,
    RotaryEmbeddings,
    create_attention_mask,
)
from merlin.models.torch.blocks.mlp import PositionwiseFeedForward
from merlin.models.torch.transforms.regularization import RMSNorm
from merlin.models.torch.utils.llama_utils import (
    convert_checkpoint,
    find_multiple,
    llama_model_lookup,
)
from merlin.models.utils.doc_utils import docstring_parameter

Self = TypeVar("Self", bound="LlamaBlock")

_LLAMA_REF = """
    ..  [1] Touvron, et al., "LLaMA: Open and Efficient Foundation Language Models".
        arXiv preprint arXiv:2302.13971 (2023).
"""


@dataclass
class LlamaConfig:
    max_seq_length: int = 2048
    vocab_size: int = 32_000
    padded_vocab_size: Optional[int] = None
    num_layers: int = 32
    num_heads: int = 32
    embedding_dim: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**LLAMA_CONFIGS[name])


LLAMA_CONFIGS = {
    "7B": dict(num_layers=32, num_heads=32, embedding_dim=4096),
    "13B": dict(num_layers=40, num_heads=40, embedding_dim=5120),
    "30B": dict(num_layers=60, num_heads=52, embedding_dim=6656),
    "65B": dict(num_layers=80, num_heads=64, embedding_dim=8192),
}


class _LlamaBaseBlock(Block):
    def __init__(
        self,
        config: LlamaConfig,
        token_key: Optional[str] = None,
        position_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert config.padded_vocab_size is not None

        self.config = config
        self.token_key = token_key or "token"
        self.position_key = position_key or "position"

    @classmethod
    def from_name(cls, model_size: str) -> Self:
        return cls(LlamaConfig.from_name(model_size))

    def reset_cache(self) -> None:
        raise NotImplementedError

    def get_tokens(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs[self.token_key]

    def get_positions(self, inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        return inputs.get(self.position_key)


@docstring_parameter(llama_reference=_LLAMA_REF)
class LlamaBlock(_LlamaBaseBlock):
    """Llama-2 [1]

    References
    ----------
    {llama_reference}
    """

    def __init__(
        self,
        config: LlamaConfig,
        token_key: Optional[str] = None,
        position_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            token_key=token_key,
            position_key=position_key,
        )
        self.transformer = LlamaTransformer(config)
        self.output_embeddings = nn.Linear(
            config.embedding_dim, config.padded_vocab_size, bias=False
        )

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        outputs = self.transformer(inputs)
        logits = self.output_embeddings(outputs)
        return logits

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir,
        model_size: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        state_dict = convert_checkpoint(checkpoint_dir, model_size)
        model_size = model_size or llama_model_lookup(state_dict)
        model = cls.from_name(model_size)
        model.load_state_dict(state_dict)
        return model

    def reset_cache(self) -> None:
        for head in self.transformer.heads:
            head.attention.kv_cache = None


class LlamaTransformer(_LlamaBaseBlock):
    def __init__(
        self,
        config: LlamaConfig,
        token_key: Optional[str] = None,
        position_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            token_key=token_key,
            position_key=position_key,
        )

        self.rotary_embeds = RotaryEmbeddings(
            self.config.embedding_dim // self.config.num_heads,
            self.config.max_seq_length,
        )
        self.mask_cache = AttentionMask(
            create_attention_mask(max_seq_length=self.config.max_seq_length)
        )

        self.token_embeddings = nn.Embedding(config.padded_vocab_size, config.embedding_dim)
        self.heads = nn.ModuleList(
            LlamaAttentionHead(
                num_heads=self.config.num_heads,
                embedding_dim=self.config.embedding_dim,
                max_seq_length=self.config.max_seq_length,
                rotary_embeds=self.rotary_embeds,
            )
            for _ in range(config.num_layers)
        )
        self.layernorm = RMSNorm(config.embedding_dim)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        if isinstance(inputs, torch.Tensor):
            tokens, positions = inputs, None
        else:
            tokens, positions = self.get_tokens(inputs), self.get_positions(inputs)

        batch_size, seq_length = tokens.size()

        x = self.token_embeddings(tokens)

        for head in self.heads:
            x = head(
                x,
                positions=positions,
                mask=self.mask_cache,
            )

        x = self.layernorm(x)

        return x

    def reset_cache(self) -> None:
        for head in self.heads:
            head.attention.kv_cache = None


class LlamaAttentionHead(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        max_seq_length: int,
        rotary_embeds: Optional[RotaryEmbeddings] = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.rotary_embeds = rotary_embeds

        self.input_layernorm = RMSNorm(self.embedding_dim)
        self.attention = CausalSelfAttention(
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
            rotary_embeds=self.rotary_embeds,
        )
        self.post_attention_layernorm = RMSNorm(self.embedding_dim)

        self.mlp = PositionwiseFeedForward(self.embedding_dim, bias=False, activation=nn.SiLU)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[AttentionMask] = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_layernorm(x),
            positions=positions,
            mask=mask,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
