import gc
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm


def llama_model_lookup(checkpoint: dict) -> str:
    """Returns the LLaMA model name from the checkpoint."""
    from merlin.models.torch.blocks.llama import LLAMA_CONFIGS

    embedding_dim = checkpoint["transformer.token_embeddings.weight"].shape[1]
    for name, configs in LLAMA_CONFIGS.items():
        if configs["embedding_dim"] == embedding_dim:
            return name

    raise RuntimeError("Could not find model name from checkpoint.")


def convert_state_dict(
    state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    converted = {}
    converted["transformer.token_embeddings.weight"] = state_dict["tok_embeddings.weight"].to(dtype)
    converted["output_embeddings.weight"] = state_dict["output.weight"].to(dtype)
    converted["transformer.layernorm.scale"] = state_dict["norm.weight"].to(dtype)

    for layer_idx in sorted(set([k.split(".")[1] for k in state_dict if k.startswith("layers")])):
        # attention
        # the wq, wk, wv from the FB model are stacked in our model.
        converted[f"transformer.heads.{layer_idx}.attention.qkv_projection.weight"] = torch.cat(
            (
                state_dict[f"layers.{layer_idx}.attention.wq.weight"].to(dtype),
                state_dict[f"layers.{layer_idx}.attention.wk.weight"].to(dtype),
                state_dict[f"layers.{layer_idx}.attention.wv.weight"].to(dtype),
            )
        )
        converted[f"transformer.heads.{layer_idx}.attention.output_projection.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wo.weight"
        ].to(dtype)
        # mlp
        converted[f"transformer.heads.{layer_idx}.mlp.weights_1.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w1.weight"
        ].to(dtype)
        converted[f"transformer.heads.{layer_idx}.mlp.projection.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w2.weight"
        ].to(dtype)
        converted[f"transformer.heads.{layer_idx}.mlp.weights_2.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w3.weight"
        ].to(dtype)
        # rms norm
        converted[f"transformer.heads.{layer_idx}.input_layernorm.scale"] = state_dict[
            f"layers.{layer_idx}.attention_norm.weight"
        ].to(dtype)
        converted[f"transformer.heads.{layer_idx}.post_attention_layernorm.scale"] = state_dict[
            f"layers.{layer_idx}.ffn_norm.weight"
        ].to(dtype)
    return converted


shard_dims = {
    "output_embeddings.weight": 0,
    "token_embeddings.weight": 1,
    "attention.qkv_projection.weight": 0,
    "attention.output_projection.weight": 1,
    "mlp.weights_1.weight": 0,
    "mlp.weights_2.weight": 0,
    "mlp.projection.weight": 1,
}


def convert_checkpoint(
    checkpoint_dir,
    model_size: str = "7B",
    dtype: str = "float32",
) -> None:
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    checkpoint_files = sorted(checkpoint_dir.glob("*.pth"))
    checkpoint_files.sort()
    n_checkpoints = len(checkpoint_files)

    if n_checkpoints == 0:
        raise RuntimeError(
            f"No checkpoints were found at checkpoint_dir {checkpoint_dir}."
            " `consolidated.0*.pth` files expected at that location."
        )

    # for the bigger models, there are multiple model-parallel checkpoints
    # and we combine them into one single file
    combined = None
    for file in tqdm(checkpoint_files, total=n_checkpoints):
        checkpoint = torch.load(file, map_location="cpu")
        converted = convert_state_dict(checkpoint, dtype=dtype)
        if combined is None:
            combined = converted
            continue
        for name, param in converted.items():
            dim = None
            for k, d in shard_dims.items():
                if k in name:
                    dim = d
                    break
            if dim is None:
                continue
            combined[name] = torch.cat((combined[name], param), dim=dim)

        del checkpoint
        del converted
        gc.collect()

    for name, param in combined.items():
        if "c_attn" not in name:
            continue

        src_chunk_len = param.shape[0] // n_checkpoints
        mat_len = src_chunk_len // 3
        dst_chunk_len = mat_len * n_checkpoints
        attn = torch.clone(param)
        for i in range(n_checkpoints):
            for j in range(3):
                param[
                    j * dst_chunk_len + i * mat_len : j * dst_chunk_len + (i + 1) * mat_len
                ] = attn[i * src_chunk_len + j * mat_len : i * src_chunk_len + (j + 1) * mat_len]

        del attn
        gc.collect()

    return combined


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)
