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
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.block import BatchBlock
from merlin.models.torch.router import RouterBlock
from merlin.models.torch.schema import LazySchemaModuleMixin, Selection, select
from merlin.schema import Schema, Tags


class TabularPadding(BatchBlock):
    """A PyTorch module for padding tabular sequence data.

    Parameters
    ----------
    schema : Schema
        The schema of the tabular data, which defines the column names of input features.
    selection : Selection
        The selection of the tabular data, which defines the column names of the
        sequence input features.
    max_sequence_length : Optional[int], default=None
        The maximum length of the sequences after padding.
        If None, sequences will be padded to the maximum length in the current batch.

    Example usage::
        features = {
            'feature1': torch.tensor([[4, 3], [5, 2]),
            'feature2': torch.tensor([[3,8], [7,9]])
        }
        schema = Schema(["feature1", "feature2"])
        _max_sequence_length = 10
        padding_op = TabularBatchPadding(
            schema=schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = padding_op(Batch(features))

    Notes:
        - If the schema contains continuous list features,
        ensure that they are normalized within the range of [0, 1].
        This is necessary because we will be padding them
        to a max_sequence_length using the minimum value of 0.0.
        - The current class only supports right padding.
    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        selection: Optional[Selection] = Tags.SEQUENCE,
        max_sequence_length: Optional[int] = None,
        name: Optional[str] = None,
    ):
        _padding = TabularPaddingModule(
            schema=schema, selection=selection, max_sequence_length=max_sequence_length
        )

        if selection is None:
            _to_add = _padding
        else:
            _to_add = RouterBlock(schema)
            _to_add.add_route(selection, _padding)

        super().__init__(_to_add, name=name)


class TabularPaddingModule(nn.Module):
    """A PyTorch module for padding tabular sequence data."""

    def __init__(
        self,
        schema: Optional[Schema] = None,
        selection: Selection = Tags.SEQUENCE,
        max_sequence_length: Optional[int] = None,
    ):
        super().__init__()
        self.selection = selection
        if schema:
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True
        self.max_sequence_length = max_sequence_length
        self.padding_idx = 0

    def initialize_from_schema(self, schema: Schema):
        self.schema = schema
        if self.selection:
            self.features: List[str] = self.schema.column_names
            self.seq_features = select(self.schema, self.selection).column_names
        else:
            self.features = self.schema.column_names
            self.seq_features = self.schema.column_names

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Batch) -> Batch:
        if not torch.jit.isinstance(inputs, Dict[str, torch.Tensor]):
            raise RuntimeError(
                "TabularPaddingModule expects a dictionary of tensors as input, ", f"got: {inputs}"
            )

        _max_sequence_length = self.max_sequence_length
        if not torch.jit.isinstance(_max_sequence_length, int):
            # Infer the maximum length from the current batch
            batch_max_sequence_length = 0
            for key, val in inputs.items():
                if key.endswith("__offsets"):
                    offsets = val
                    max_row_length = int(torch.max(offsets[1:] - offsets[:-1]))
                    batch_max_sequence_length = max(max_row_length, batch_max_sequence_length)
            _max_sequence_length = batch_max_sequence_length

        # Store the non-padded lengths of list features
        seq_inputs_lengths = self._get_sequence_lengths(inputs)
        seq_shapes: List[torch.Tensor] = list(seq_inputs_lengths.values())
        if not torch.all(torch.stack([torch.all(x == seq_shapes[0]) for x in seq_shapes])):
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

        return Batch(
            features=batch_padded, targets=targets_padded, sequences=Sequence(seq_inputs_lengths)
        )

    def _get_sequence_lengths(self, sequences: Dict[str, torch.Tensor]):
        """Compute the effective length of each sequence in a dictionary of sequences."""
        seq_inputs_lengths = {}
        for key, val in sequences.items():
            if key.endswith("__offsets"):
                seq_inputs_lengths[key[: -len("__offsets")]] = val[1:] - val[:-1]
            elif key in self.seq_features:
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

    def _pad_dense_tensor(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        """Pad a dense tensor along its second dimension to a specified length."""
        if len(tensor.shape) == 2:
            pad_diff = length - tensor.shape[1]
            return F.pad(input=tensor, pad=(0, pad_diff, 0, 0))
        return tensor


class BroadcastToSequence(nn.Module, LazySchemaModuleMixin):
    """
    A PyTorch module to broadcast features to match the sequence length.

    BroadcastToSequence is a PyTorch module designed to facilitate broadcasting
    of specific features within a given data schema to match a given sequence length.
    This can be particularly useful in sequence-based neural networks, where different
    types of inputs need to be processed in sync within the network, and all inputs need
    to be of the same length.

    For example, in a sequence-to-sequence learning problem, one might have a feature
    representing a constant property for each sequence (like an ID or a group), and you
    want this feature to be available at each time step. In this case, you can use
    BroadcastToSequence to 'broadcast' this feature along the time dimension,
    creating a copy for each time step.

    Parameters
    ----------
    to_broadcast : Selection
        The features that need to be broadcasted.
    sequence : Selection
        The sequence features.

    """

    def __init__(
        self,
        to_broadcast: Selection,
        sequence: Selection,
        schema: Optional[Schema] = None,
    ):
        super().__init__()
        self.to_broadcast = to_broadcast
        self.sequence = sequence
        self.to_broadcast_features: List[str] = []
        self.sequence_features: List[str] = []
        if schema:
            self.initialize_from_schema(schema)

    def initialize_from_schema(self, schema: Schema):
        """
        Initialize the module from a schema.

        Parameters
        ----------
        schema : Schema
            The input-schema of this module
        """
        super().initialize_from_schema(schema)
        self.schema = schema
        self.to_broadcast_features = select(schema, self.to_broadcast).column_names
        self.sequence_features = select(schema, self.sequence).column_names

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward propagation method.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            The inputs dictionary containing the tensors to be broadcasted.

        Returns
        -------
        Dict[str, torch.Tensor]
            The dictionary containing the broadcasted tensors.

        Raises
        ------
        RuntimeError
            If a tensor has an unsupported number of dimensions.
        """

        outputs = {}
        seq_length = self.get_seq_length(inputs)

        # Iterate over the to_broadcast_features and broadcast each tensor to the sequence length
        for key, val in inputs.items():
            if key in self.to_broadcast_features:
                # Check the dimension of the original tensor
                if len(val.shape) == 1:  # for 1D tensor (batch dimension only)
                    broadcasted_tensor = val.unsqueeze(1).repeat(1, seq_length)
                elif len(val.shape) == 2:  # for 2D tensor (batch dimension + feature dimension)
                    broadcasted_tensor = val.unsqueeze(1).repeat(1, seq_length, 1)
                else:
                    raise RuntimeError(f"Unsupported number of dimensions: {len(val.shape)}")

                # Update the inputs dictionary with the broadcasted tensor
                outputs[key] = broadcasted_tensor
            else:
                outputs[key] = val

        return outputs

    def get_seq_length(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Get the sequence length from inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            The inputs dictionary.

        Returns
        -------
        int
            The sequence length.
        """

        if len(self.sequence_features) == 0:
            raise RuntimeError("No sequence features found in the inputs.")

        first_feat: str = self.sequence_features[0]

        if first_feat + "__offsets" in inputs:
            return inputs[first_feat + "__offsets"][-1].item()

        return inputs[first_feat].shape[1]


class TabularPredictNext(BatchBlock):
    """A Batchblock instance for preparing sequential inputs and targets
    for next-item prediction. The target is extracted from the shifted
    sequence of the target feature and the sequential input features
    are truncated in the last position.

    Parameters
    ----------
    target : Optional[Selection], default=Tags.ID
        The sequential input column(s) that will be used to extract the target.
        Targets can be one or multiple input features with the same sequence length.
    schema : Optional[Schema]
        The schema with the sequential columns to be truncated
    apply_padding : Optional[bool], default=True
        Whether to pad sequential inputs before extracting the target(s).
    max_sequence_length : Optional[int], default=None
        The maximum length of the sequences after padding.
        If None, sequences will be padded to the maximum length in the current batch.

    Example usage::
        batch_output = transform(batch)

        features = {
            'feature1': torch.tensor([[4, 3], [5, 2]),
            'feature2': torch.tensor([[3,8], [7,9]])
        }
        schema = Schema(["feature1", "feature2"])
        next_item_op = TabularPredictNext(
            schema=schema, target='feature1'
        )
        transformed_batch = next_item_op(Batch(features))
    """

    def __init__(
        self,
        target: Selection = Tags.ID,
        schema: Optional[Schema] = None,
        apply_padding: bool = True,
        max_sequence_length: int = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            TabularPredictNextModule(
                schema=schema,
                target=target,
                apply_padding=apply_padding,
                max_sequence_length=max_sequence_length,
            ),
            name=name,
        )


class TabularSequenceTransform(nn.Module):
    """Base PyTorch module for preparing targets from a batch of sequential inputs.
    Parameters
    ----------
    target : Optional[Selection], default=Tags.ID
        The sequential input column that will be used to extract the target.
        In case of multiple targets, either a list of target feature names
        or a shared Tag indicating the targets should be provided.
    schema : Optional[Schema]
        The schema with the sequential columns to be truncated
    apply_padding : Optional[bool], default=True
        Whether to pad sequential inputs before extracting the target(s).
    max_sequence_length : Optional[int], default=None
        The maximum length of the sequences after padding.
        If None, sequences will be padded to the maximum length in the current batch.
    """

    def __init__(
        self,
        target: Optional[Selection] = Tags.ID,
        schema: Optional[Schema] = None,
        apply_padding: bool = True,
        max_sequence_length: int = None,
    ):
        super().__init__()
        self.target = target
        if schema:
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True
        self.padding_idx = 0
        self.apply_padding = apply_padding
        if self.apply_padding:
            self.padding_operator = TabularPadding(
                schema=self.schema, max_sequence_length=max_sequence_length
            )

    def initialize_from_schema(self, schema: Schema):
        self.schema = schema
        self.features: List[str] = self.schema.column_names
        target = select(self.schema, self.target)
        if not target:
            raise ValueError(
                f"The target '{self.target}' was not found in the "
                f"provided sequential schema: {self.schema}"
            )
        self.target_name = self._get_target(target)

    def _get_target(self, target):
        return target.column_names

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Batch, **kwargs
    ) -> Batch:
        raise NotImplementedError()

    def _check_seq_inputs_targets(self, batch: Batch):
        self._check_input_sequence_lengths(batch)
        self._check_target_shape(batch)

    def _check_target_shape(self, batch: Batch):
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

    def _check_input_sequence_lengths(self, batch: Batch):
        if not batch.sequences.lengths:
            raise ValueError(
                "The input `batch` should include information about input sequences lengths"
            )
        sequence_lengths = torch.stack([batch.sequences.length(name) for name in self.features])
        assert torch.all(sequence_lengths.eq(sequence_lengths[0])), (
            "All tabular sequence features need to have the same sequence length, "
            f"found {sequence_lengths}"
        )


class TabularPredictNextModule(TabularSequenceTransform):
    """A PyTorch module for preparing tabular sequence data for next-item prediction."""

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Batch) -> Batch:
        if self.apply_padding:
            batch = self.padding_operator(batch)
        self._check_seq_inputs_targets(batch)

        # Shifts the target column to be the next item of corresponding input column
        new_targets: Dict[str, torch.Tensor] = dict()
        for name in self.target_name:
            new_target = batch.features[name]
            new_target = new_target[:, 1:]
            new_targets[name] = new_target

        # Removes the last item of the sequence, as it belongs to the target
        new_inputs = dict()
        for k, v in batch.features.items():
            if k in self.features:
                new_inputs[k] = v[:, :-1]

        # Generates information about the new lengths and causal masks
        new_lengths, causal_masks = {}, {}
        for name in self.features:
            new_lengths[name] = batch.sequences.lengths[name] - 1
        _max_length = list(new_targets.values())[0].shape[
            -1
        ]  # all new targets have same output sequence length
        causal_mask = self._generate_causal_mask(list(new_lengths.values())[0], _max_length)
        for name in self.features:
            causal_masks[name] = causal_mask

        return Batch(
            features=new_inputs,
            targets=new_targets,
            sequences=Sequence(new_lengths, masks=causal_masks),
        )

    def _generate_causal_mask(self, seq_lengths: torch.Tensor, max_len: int):
        """
        Generate a 2D mask from a tensor of sequence lengths.
        """
        return torch.arange(max_len)[None, :] < seq_lengths[:, None]
