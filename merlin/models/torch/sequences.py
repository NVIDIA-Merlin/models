#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.schema import Selection, select
from merlin.schema import ColumnSchema, Schema, Tags

MASK_PREFIX = "__mask"


class TabularPadding(nn.Module):
    """A PyTorch module for padding tabular sequence data.

    Parameters
    ----------
    schema : Schema
        The schema of the tabular data, which defines the column names of input features.
    max_sequence_length : Optional[int], default=None
        The maximum length of the sequences after padding.
        If None, sequences will be padded to the maximum length in the current batch.


    Examples:
        features = {
            'feature1': torch.tensor([[4, 3], [5, 2]),
            'feature2': torch.tensor([[3,8], [7,9]])
        }
        schema = Schema(["feature1", "feature2"])
        _max_sequence_length = 10
        padding_op = TabularBatchPadding(
            schema=schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = padding_op(Batch(feaures))

    Note:
        If the schema contains continuous list features,
        ensure that they are normalized within the range of [0, 1].
        This is necessary because we will be padding them
        to a max_sequence_length using the minimum value of 0.0.
    """

    def __init__(
        self,
        schema: Schema,
        max_sequence_length: Optional[int] = None,
    ):
        super().__init__()
        self.schema = schema
        self.max_sequence_length = max_sequence_length
        self.features: List[str] = self.schema.column_names
        self.sparse_features = self.schema.select_by_tag(Tags.SEQUENCE).column_names
        self.padding_idx = 0

    def forward(self, batch: Batch) -> Batch:
        _max_sequence_length = self.max_sequence_length
        if not _max_sequence_length:
            # Infer the maximum length from the current batch
            batch_max_sequence_length = 0
            for key, val in batch.features.items():
                if key.endswith("__offsets"):
                    offsets = val
                    max_row_length = int(torch.max(offsets[1:] - offsets[:-1]))
                    batch_max_sequence_length = max(max_row_length, batch_max_sequence_length)
            _max_sequence_length = batch_max_sequence_length

        # Store the non-padded lengths of list features
        seq_inputs_lengths = self._get_sequence_lengths(batch.features)
        seq_shapes = list(seq_inputs_lengths.values())
        if not all(torch.all(x == seq_shapes[0]) for x in seq_shapes):
            raise ValueError(
                "The sequential inputs must have the same length for each row in the batch, "
                f"but they are different: {seq_shapes}"
            )

        # Pad the features of the batch
        batch_padded = {}
        for key, value in batch.features.items():
            if key.endswith("__offsets"):
                col_name = key[: -len("__offsets")]
                if col_name in self.features:
                    padded_values = self._pad_ragged_tensor(
                        batch.features[f"{col_name}__values"], value, _max_sequence_length
                    )
                    batch_padded[col_name] = padded_values
            elif key.endswith("__values"):
                continue
            else:
                col_name = key
                if col_name in self.features and seq_inputs_lengths.get(col_name) is not None:
                    # pad dense list features
                    batch_padded[col_name] = self._pad_dense_tensor(value, _max_sequence_length)

        # Pad targets of the batch
        targets_padded = None
        if batch.targets is not None:
            targets_padded = {}
            for key, value in batch.targets.items():
                if key.endswith("__offsets"):
                    col_name = key[: -len("__offsets")]
                    padded_values = self._pad_ragged_tensor(
                        batch.targets[f"{col_name}__values"], value, _max_sequence_length
                    )
                    targets_padded[col_name] = padded_values
                elif key.endswith("__values"):
                    continue
                else:
                    targets_padded[key] = value
        # TODO: do we store lengths of sequential targets features too?
        return Batch(
            features=batch_padded, targets=targets_padded, sequences=Sequence(seq_inputs_lengths)
        )

    def _get_sequence_lengths(self, sequences: Dict[str, torch.Tensor]):
        """Compute the effective length of each sequence in a dictionary of sequences."""
        seq_inputs_lengths = {}
        for key, val in sequences.items():
            if key.endswith("__offsets"):
                seq_inputs_lengths[key[: -len("__offsets")]] = val[1:] - val[:-1]
            elif key in self.sparse_features:
                seq_inputs_lengths[key] = (val != self.padding_idx).sum(-1)
        return seq_inputs_lengths

    def _squeeze(self, tensor: torch.Tensor):
        """Squeeze a tensor of shape (N,1) to shape (N)."""
        if len(tensor.shape) == 2:
            return tensor.squeeze(1)
        return tensor

    def _get_indices(self, offsets: torch.Tensor, diff_offsets: torch.Tensor):
        """Compute indices for a sparse tensor from offsets and their differences."""
        row_ids = torch.arange(len(offsets) - 1, device=offsets.device)
        row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
        row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
        col_ids = (
            torch.arange(len(row_offset_repeated), device=offsets.device) - row_offset_repeated
        )
        indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], dim=1)
        return indices

    def _pad_ragged_tensor(self, values: torch.Tensor, offsets: torch.Tensor, padding_length: int):
        """Pad a ragged features represented by "values" and "offsets" to a dense tensor
        of length `padding_length`.
        """
        values = self._squeeze(values)
        offsets = self._squeeze(offsets)
        num_rows = len(offsets) - 1
        diff_offsets = offsets[1:] - offsets[:-1]
        max_length = int(diff_offsets.max())
        indices = self._get_indices(offsets, diff_offsets)
        sparse_tensor = torch.sparse_coo_tensor(
            indices.T, values, torch.Size([num_rows, max_length]), device=values.device
        )

        return self._pad_dense_tensor(sparse_tensor.to_dense(), padding_length)

    def _pad_dense_tensor(self, t: torch.Tensor, length: int) -> torch.Tensor:
        """Pad a dense tensor along its second dimension to a specified length."""
        if len(t.shape) == 2:
            pad_diff = length - t.shape[1]
            return F.pad(input=t, pad=(0, pad_diff, 0, 0))
        return t


class TabularSequenceTransform(nn.Module):
    """Base class for preparing targets from a batch of sequential inputs.
    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema, Schema]
        The sequential input column that will be used to extract the target.
        For multiple targets usecase, one should provide a Schema containing
        all target columns.
    """

    def __init__(
        self,
        schema: Schema,
        target: Selection,
        apply_padding: bool = True,
        max_sequence_length: int = None,
    ):
        super().__init__()
        self.schema = schema
        self.features = schema.column_names
        self.target = select(self.schema, target)
        self.target_name = self._get_target(self.target)
        self.padding_idx = 0
        self.apply_padding = apply_padding
        if self.apply_padding:
            self.padding_operator = TabularPadding(
                schema=self.schema, max_sequence_length=max_sequence_length
            )

    def _get_target(self, target):
        return target.column_names

    def forward(self, batch: Batch, **kwargs) -> Tuple:
        raise NotImplementedError()

    def _check_seq_inputs_targets(self, batch: Batch):
        self._check_input_sequence_lengths(batch)
        self._check_target_shape(batch)

    def _check_target_shape(self, batch):
        for name in self.target_name:
            if name not in batch.features:
                raise ValueError(f"Inputs features do not contain target column ({name})")

            target = batch.features[name]
            if target.ndim < 2:
                raise ValueError(
                    f"Sequential target column ({name}) "
                    f"must be a 2D tensor, but shape is {target.ndim}"
                )
            lengths = batch.sequences.length(name)
            if any(lengths <= 1):
                raise ValueError(
                    f"2nd dim of target column ({name})"
                    "must be greater than 1 for sequential input to be shifted as target"
                )

    def _check_input_sequence_lengths(self, batch):
        if batch.sequences is None:
            raise ValueError(
                "The input `batch` should include information about input sequences lengths"
            )
        sequence_lengths = torch.stack([batch.sequences.length(name) for name in self.features])
        assert torch.all(sequence_lengths.eq(sequence_lengths[0])), (
            "All tabular sequence features need to have the same sequence length, "
            f"found {sequence_lengths}"
        )


class TabularPredictNext(TabularSequenceTransform):
    """Prepares sequential inputs and targets for next-item prediction.
    The target is extracted from the shifted sequence of the target feature and
    the sequential input features are truncated in the last position.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, List[str], Tags, ColumnSchema, Schema]
        The sequential input column(s) that will be used to extract the target.
        Targets can be one or multiple input features with the same sequence length.

    Examples:
        transform = TabularPredictNext(
            schema=schema.select_by_tag(Tags.SEQUENCE), target="a"
        )
        batch_output = transform(batch)

    """

    def _generate_causal_mask(self, seq_lengths, max_len):
        """
        Generate a 2D mask from a tensor of sequence lengths.
        """
        return torch.arange(max_len)[None, :] < seq_lengths[:, None]

    def forward(self, batch: Batch, **kwargs) -> Batch:
        if self.apply_padding:
            batch = self.padding_operator(batch)
        self._check_seq_inputs_targets(batch)
        # Shifts the target column to be the next item of corresponding input column
        new_targets = {}
        for name in self.target_name:
            new_target = batch.features[name]
            new_target = new_target[:, 1:]
            new_targets[name] = new_target

        # Removes the last item of the sequence, as it belongs to the target
        new_inputs = dict()
        for k, v in batch.features.items():
            if k in self.features:
                new_inputs[k] = v[:, :-1]

        # Generates information about new lengths and causal masks
        new_lengths, causal_masks = {}, {}
        _max_length = new_target.shape[-1]  # all new targets have same output sequence length
        for name in self.features:
            new_lengths[name] = batch.sequences.lengths[name] - 1
            causal_masks[name] = self._generate_causal_mask(new_lengths[name], _max_length)

        return Batch(
            features=new_inputs,
            targets=new_targets,
            sequences=Sequence(new_lengths, masks=causal_masks),
        )


class TabularMaskRandom(TabularSequenceTransform):
    """This transform implements the Masked Language Modeling (MLM) training approach
    introduced in BERT (NLP) and later adapted to RecSys by BERT4Rec [1].
    Given an input `Batch` with input features including the sequence of candidates ids,
    some positions are randomly selected (masked) to be the targets for prediction.
    The targets are output being the same as the input candidates ids sequence.
    The target masks are returned within the `Bathc.Sequence` object.

    References
    ----------
    .. [1] Sun, Fei, et al. "BERT4Rec: Sequential recommendation with bidirectional encoder
           representations from transformer." Proceedings of the 28th ACM international
           conference on information and knowledge management. 2019.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential inputs to be masked
    target : Union[str, List[str], Tags, ColumnSchema, Schema]
        The sequential input column(s) that will be used to compute the masked positions.
        Targets can be one or multiple input features with the same sequence length.
    masking_prob : float, optional
        Probability of a candidate to be selected (masked) as a label of the given sequence.
        Note: We enforce that at least one candidate is masked for each sequence, so that it
        is useful for training, by default 0.2

    Examples:
        transform = TabularMaskRandom(
            schema=schema.select_by_tag(Tags.SEQUENCE), target="a", masking_prob=0.4
        )
        batch_output = transform(batch)

    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        masking_prob: float = 0.2,
        **kwargs,
    ):
        self.masking_prob = masking_prob
        super().__init__(schema, target, **kwargs)

    def forward(self, batch: Batch, **kwargs) -> Tuple:
        if self.apply_padding:
            batch = self.padding_operator(batch)
        self._check_seq_inputs_targets(batch)
        new_targets = dict({name: torch.clone(batch.features[name]) for name in self.target_name})
        new_inputs = {feat: batch.features[feat] for feat in self.features}
        sequence_lengths = {feat: batch.sequences.length(feat) for feat in self.features}

        # Generates mask information for the group of input sequences
        target_mask = self._generate_mask(new_targets[self.target_name[0]])
        random_mask = {name: target_mask for name in self.features}

        return Batch(
            features=new_inputs,
            targets=new_targets,
            sequences=Sequence(sequence_lengths, masks=random_mask),
        )

    def _generate_mask(self, new_target: torch.Tensor) -> torch.Tensor:
        """Generate mask information at random positions from a 2D target sequence"""

        non_padded_mask = new_target != self.padding_idx
        rows_ids = torch.arange(new_target.size(0), dtype=torch.long, device=new_target.device)

        # 1. Selects a percentage of non-padded candidates to be masked (selected as targets)
        probability_matrix = torch.full(
            new_target.shape, self.masking_prob, device=new_target.device
        )
        mask_targets = torch.bernoulli(probability_matrix).bool() & non_padded_mask

        # 2. Set at least one candidate in the sequence to mask, so that the network
        # can learn something with this session
        one_random_index_by_row = torch.multinomial(
            non_padded_mask.float(), num_samples=1
        ).squeeze()
        mask_targets[rows_ids, one_random_index_by_row] = True

        # 3. If a sequence has only masked targets, unmasks one of the targets
        sequences_with_only_labels = mask_targets.sum(dim=1) == non_padded_mask.sum(dim=1)
        sampled_targets_to_unmask = torch.multinomial(mask_targets.float(), num_samples=1).squeeze()
        targets_to_unmask = torch.masked_select(
            sampled_targets_to_unmask, sequences_with_only_labels
        )
        rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)
        mask_targets[rows_to_unmask, targets_to_unmask] = False
        return mask_targets


class TabularMaskLast(TabularSequenceTransform):
    """This transform copies one of the sequence input features to be
    the target feature. The last item of the target sequence is selected (masked)
    to be predicted.
    The target masks are returned by copying the related input features.


    Parameters
    ----------
    schema : Schema
        The schema with the sequential inputs to be masked
    target : Union[str, List[str], Tags, ColumnSchema, Schema]
        The sequential input column(s) that will be used to compute the masked positions.
        Targets can be one or multiple input features with the same sequence length.

    Examples:
        transform = TabularMaskLast(
            schema=schema.select_by_tag(Tags.SEQUENCE), target="a"
        )
        batch_output = transform(batch)

    """

    def forward(self, batch: Batch, **kwargs) -> Tuple:
        if self.apply_padding:
            batch = self.padding_operator(batch)
        self._check_seq_inputs_targets(batch)
        new_targets = dict({name: torch.clone(batch.features[name]) for name in self.target_name})
        new_inputs = {feat: batch.features[feat] for feat in self.features}
        sequence_lengths = {feat: batch.sequences.length(feat) for feat in self.features}

        # Generates mask information for the group of input sequences
        target_mask = self._generate_mask(new_targets[self.target_name[0]])
        masks = {name: target_mask for name in self.features}

        return Batch(
            features=new_inputs,
            targets=new_targets,
            sequences=Sequence(sequence_lengths, masks=masks),
        )

    def _generate_mask(self, new_target: torch.Tensor) -> torch.Tensor:
        """Generate mask information at last positions from a 2D target sequence"""
        target_mask = new_target != self.padding_idx
        last_non_padded_indices = (target_mask.sum(dim=1) - 1).unsqueeze(-1)

        mask_targets = (
            torch.arange(target_mask.size(1), device=target_mask.device).unsqueeze(0)
            == last_non_padded_indices
        )
        return mask_targets
