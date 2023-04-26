from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def get_last(padded_tensor: torch.Tensor) -> torch.Tensor:
    # Find the length of each sequence in the batch
    sequence_lengths = (padded_tensor != 0).sum(dim=1)

    # Calculate the indices of the last valid elements
    last_element_indices = (sequence_lengths - 1).unsqueeze(-1)

    # Gather the last elements from the padded tensor
    last_elements = torch.gather(padded_tensor, 1, last_element_indices).squeeze()

    return last_elements


def pop_last(padded_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find the length of each sequence in the batch
    sequence_lengths = (padded_tensor != 0).sum(dim=1)

    # Calculate the indices of the last valid elements
    last_element_indices = (sequence_lengths - 1).unsqueeze(-1)

    # Gather the last elements from the padded tensor
    last_elements = torch.gather(padded_tensor, 1, last_element_indices).squeeze()

    # Create a mask to remove the last valid elements from the padded tensor
    mask = torch.arange(padded_tensor.size(1)).unsqueeze(0) < last_element_indices
    transformed_padded_tensor = padded_tensor * mask

    return transformed_padded_tensor, last_elements


def remove_last_non_padded(padded_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    first_tensor = next(iter(padded_tensors.values()))
    sequence_lengths = (first_tensor != 0).sum(dim=1)
    last_element_indices = (sequence_lengths - 1).unsqueeze(-1)

    outputs = {}
    for key, tensor in padded_tensors.items():
        mask = torch.arange(tensor.size(1)).unsqueeze(0) < last_element_indices

        outputs[key] = (tensor * mask)[:, :-1]

    return outputs


def _pad_dense_tensor(t: torch.Tensor, length: int) -> torch.Tensor:
    if len(t.shape) == 2:
        pad_diff = length - t.shape[1]
        return F.pad(input=t, pad=(0, pad_diff, 0, 0))
    return t


def _squeeze(tensor: torch.Tensor):
    if len(tensor.shape) == 2:
        return tensor.squeeze(1)
    return tensor


def _get_indices(offsets: torch.Tensor, diff_offsets: torch.Tensor):
    row_ids = torch.arange(len(offsets) - 1, device=offsets.device)
    row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
    col_ids = torch.arange(len(row_offset_repeated), device=offsets.device) - row_offset_repeated
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], dim=1)
    return indices


def _pad_ragged_tensor(values: torch.Tensor, offsets: torch.Tensor, padding_length: int):
    values = _squeeze(values)
    offsets = _squeeze(offsets)
    num_rows = len(offsets) - 1
    diff_offsets = offsets[1:] - offsets[:-1]
    max_length = int(diff_offsets.max())
    indices = _get_indices(offsets, diff_offsets)
    sparse_tensor = torch.sparse_coo_tensor(
        indices.T, values, torch.Size([num_rows, max_length]), device=values.device
    )
    return _pad_dense_tensor(sparse_tensor.to_dense(), padding_length)


@torch.jit.script
def pad_batch(
    batch: Dict[str, torch.Tensor], padding_lengths: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """Pad list features in a batch to padding length specified

    Parameters
    ----------
    batch : Batch
        dictionary of tensors in batch
    padding_lengths : Dict[str, int]
        dictionary mapping list column name to padding length

    Returns
    -------
    Batch
        Batch with padded list features

    Raises
    ------
    ValueError
        If ragged column found with no padding length provided
    """
    batch_padded = {}
    for key, value in batch.items():
        if key.endswith("__offsets"):
            col_name = key[: -len("__offsets")]
            padding_length = padding_lengths.get(col_name)
            if padding_length is not None:
                padded_values = _pad_ragged_tensor(
                    batch[f"{col_name}__values"], value, padding_length
                )
                batch_padded[col_name] = padded_values
            else:
                # Note: This exception can be removed if the model is
                # updated to support __values / __offsets inputs
                raise ValueError(
                    f"Found ragged column '{col_name}' with unspecified padding length. "
                    "Please provide a padding length for this feature "
                    "to be converted to a dense tensor. "
                )
        elif key.endswith("__values"):
            continue
        else:
            col_name = key
            padding_length = padding_lengths.get(col_name)
            if padding_length is not None:
                batch_padded[col_name] = _pad_dense_tensor(value, padding_length)
            else:
                batch_padded[col_name] = value

    return batch_padded


@torch.jit.script
def pad_inputs(inputs: Dict[str, torch.Tensor], max_sequence_length: Optional[int] = None):
    """Pad ragged inputs to fixed size tensors.

    Pads all the sequence features in the inputs to the same length.
    The minimum of max_sequence_length and the maximum sequence length in the inputs.

    Parameters
    ----------
    inputs : Dict[str, Tensor]
        Dictionary of tensors
    max_sequence_length: int, optional
        The maximum sequence length to limit sequence features to

    Returns
    -------
    inputs : Dict[str, Tensor]
        Padded inputs
    """
    batch_max_sequence_length = 0
    seq_lenghts: Dict[str, torch.Tensor] = {}
    for key, val in inputs.items():
        if key.endswith("__offsets"):
            offsets = val
            lengths = offsets[1:] - offsets[:-1]
            max_row_length = int(torch.max(lengths))
            batch_max_sequence_length = max(max_row_length, batch_max_sequence_length)
            seq_lenghts[key[: -len("__offsets")]] = lengths

    padding_sequence_length = batch_max_sequence_length
    if max_sequence_length is not None:
        padding_sequence_length = min(max_sequence_length, batch_max_sequence_length)

    if padding_sequence_length > 0:
        padding_lengths: Dict[str, int] = {}
        for key in inputs.keys():
            if key.endswith("__offsets"):
                col_name: str = key[: -len("__offsets")]
                padding_lengths[col_name] = padding_sequence_length
        if padding_lengths:
            inputs = pad_batch(inputs, padding_lengths)

    return inputs, seq_lenghts
