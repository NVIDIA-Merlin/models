from copy import deepcopy
from typing import Dict, Optional, Union

import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.utils.doc_utils import docstring_parameter

_ROPE_REF = """
    ..  [1] Su, et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
        arXiv preprint arXiv:2104.09864 (2021).
"""
_TRANSFORMER_REF = """
    ..  [1] Vaswani, et al., "Attention Is All You Need".
        arXiv preprint arXiv:1706.03762 (2017).
"""


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


@docstring_parameter(rope_reference=_ROPE_REF)
class RotaryEmbeddings(nn.Module):
    """Rotary Position Embedding (RoPE) as proposed in [1].

    References
    ----------
    {rope_reference}
    """

    def __init__(self, dim: int, max_seq_length: int, base: int = 10000) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.dim = dim
        self.base = base

        self.cache = None

    def initialize(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        inverse_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=dtype, device=device) / self.dim)
        )
        self.register_buffer("inverse_freq", inverse_freq, persistent=False)

        position = torch.arange(self.max_seq_length, dtype=dtype, device=device)
        freq = torch.outer(position, self.inverse_freq).float()
        cache = torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.half()

        self.cache = cache
        self._is_initialized = True

    def forward(
        self,
        inputs: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self.cache is None:
            self.initialize(device=device, dtype=dtype)

        batch_size, seq_length, width, _ = inputs.size()

        if positions is not None:
            _cache = self.cache.index_select(0, positions)
        else:
            _cache = self.cache[:seq_length]

        _inputs = inputs.float().reshape(batch_size, seq_length, width, -1, 2)
        _cache = _cache.view(1, _inputs.size(1), 1, _inputs.size(3), 2)
        outputs = torch.stack(
            [
                _inputs[..., 0] * _cache[..., 0] - _inputs[..., 1] * _cache[..., 1],
                _inputs[..., 1] * _cache[..., 0] + _inputs[..., 0] * _cache[..., 1],
            ],
            -1,
        )

        return outputs.flatten(3).type_as(inputs)


@torch.jit.script
class AttentionMask:
    def __init__(self, bool_mask: torch.Tensor) -> None:
        self.bool_mask = bool_mask

    def select(self, seq_length: int) -> torch.Tensor:
        return self.bool_mask[:, :, :seq_length, :seq_length]

    def select_position(self, position: torch.Tensor) -> torch.Tensor:
        return self.bool_mask.index_select(2, position)


def create_attention_mask(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones(
        (max_seq_length, max_seq_length),
        device=device,
        dtype=torch.bool,
    )
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


@torch.jit.script
class KeyValueCache:
    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.key = key
        self.value = value

    def cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        dim: int = -2,
        max_seq_length: Optional[int] = None,
    ):
        if max_seq_length is None:
            max_seq_length = key.size(dim)

        cached_key, cached_value = self.key, self.value
        # check if reached token limit
        if positions[-1] >= max_seq_length:
            positions = torch.tensor(max_seq_length - 1, device=positions.device)
            # shift 1 position to the left
            cached_key = torch.roll(cached_key, -1, dims=dim)
            cached_value = torch.roll(cached_value, -1, dims=dim)
        key = cached_key.index_copy(dim, positions, key)
        value = cached_value.index_copy(dim, positions, value)

        self.key, self.value = key, value

        return key, value


@docstring_parameter(
    transformer_reference=_TRANSFORMER_REF, rope_reference=_ROPE_REF.replace("[1]", "[2]")
)
class CausalSelfAttention(nn.Module):
    """Transformer self-attention [1].

    The key difference between our implementation and PyTorch implemention,
    i.e., ``torch.nn.MultiheadAttention`` is that Rotary Position Embedding [2]
    is applied to the key and query matrices. ``torch.nn.MultiheadAttention``
    is currently too rigid to support such variation.

    References
    ----------
    {transformer_reference}
    {rope_reference}
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        max_seq_length: int,
        bias: bool = False,
        dropout_p: float = 0.0,
        store_cache: bool = True,
        kv_cache: Optional[KeyValueCache] = None,
        rotary_embeds: Optional[RotaryEmbeddings] = None,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "The embedding dimension must be divible by the number of self-attention heads"
            )

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.bias = bias
        self.dropout_p = dropout_p
        self.store_cache = store_cache
        self.kv_cache = kv_cache
        self.rotary_embeds = rotary_embeds

        # query, key, and value projections for all heads, but in a batch.
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim, bias=self.bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=self.bias)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[AttentionMask] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, embedding_dim = x.size()

        if self.store_cache and self.kv_cache is None:
            head_size = self.embedding_dim // self.num_heads
            cache_shape = (batch_size, self.num_heads, self.max_seq_length, head_size)
            self.kv_cache = KeyValueCache(
                key=torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                value=torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
            )

        # calculate query, key, values for all heads in batch
        # and move head forward to be the batch dim
        q, k, v = self.qkv_projection(x).split(self.embedding_dim, dim=2)

        head_size = embedding_dim // self.num_heads
        k = k.view(batch_size, seq_length, self.num_heads, head_size)
        q = q.view(batch_size, seq_length, self.num_heads, head_size)
        v = v.view(batch_size, seq_length, self.num_heads, head_size)

        if self.rotary_embeds is not None:
            q = self.rotary_embeds(q, positions)
            k = self.rotary_embeds(k, positions)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.kv_cache is not None and positions is not None:
            k, v = self.kv_cache.cache(
                key=k,
                value=v,
                positions=positions,
                max_seq_length=self.max_seq_length,
            )

        if mask is not None:
            if positions is not None:
                attn_mask = mask.select_position(positions)
            else:
                attn_mask = mask.select(seq_length)
        else:
            attn_mask = None

        # efficient attention using Flash Attention CUDA kernels
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, embedding_dim)

        y = self.output_projection(y)

        return y
