import re
from itertools import accumulate

import pytest
import torch

from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.sequences import (
    TabularBatchPadding,
    TabularMaskRandom,
    TabularPredictNext,
    TabularSequenceTransform,
)
from merlin.schema import ColumnSchema, Schema, Tags


def _get_values_offsets(data):
    values = []
    row_lengths = []
    for row in data:
        row_lengths.append(len(row))
        values += row
    offsets = [0] + list(accumulate(row_lengths))
    return torch.tensor(values), torch.tensor(offsets)


class TestPadBatch:
    @pytest.fixture
    def sequence_batch(self):
        a_values, a_offsets = _get_values_offsets(data=[[1, 2], [], [3, 4, 5]])
        b_values, b_offsets = _get_values_offsets([[34, 30], [], [33, 23, 50]])
        features = {
            "a__values": a_values,
            "a__offsets": a_offsets,
            "b__values": b_values,
            "b__offsets": b_offsets,
            "c_dense": torch.Tensor([[1, 2, 0], [0, 0, 0], [4, 5, 6]]),
            "d_context": torch.Tensor([1, 2, 3]),
        }
        targets = None
        return Batch(features, targets)

    @pytest.fixture
    def sequence_schema(self):
        return Schema(
            [
                ColumnSchema("a", tags=[Tags.SEQUENCE]),
                ColumnSchema("b", tags=[Tags.SEQUENCE]),
                ColumnSchema("c_dense", tags=[Tags.SEQUENCE]),
                ColumnSchema("d_context", tags=[Tags.CONTEXT]),
            ]
        )

    def test_padded_features(self, sequence_batch, sequence_schema):
        _max_sequence_length = 8
        padding_op = TabularBatchPadding(
            schema=sequence_schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = padding_op(sequence_batch)

        assert torch.equal(padded_batch.sequences.length("a"), torch.Tensor([2, 0, 3]))
        assert set(padded_batch.features.keys()) == set(sequence_schema.column_names)
        for feature in ["a", "b", "c_dense"]:
            assert padded_batch.features[feature].shape[1] == _max_sequence_length
        assert torch.equal(padded_batch.features["d_context"], sequence_batch.features["d_context"])

    def test_batch_invalid_lengths(self):
        # Test when targets is not a tensor nor a dictionary of tensors
        a_values, a_offsets = _get_values_offsets(data=[[1, 2], [], [3, 4, 5]])
        b_values, b_offsets = _get_values_offsets([[34], [23, 56], [33, 23, 50, 4]])

        with pytest.raises(
            ValueError,
            match="The sequential inputs must have the same length for each row in the batch",
        ):
            padding_op = TabularBatchPadding(schema=Schema(["a", "b"]))
            padding_op(
                Batch(
                    {
                        "a__values": a_values,
                        "a__offsets": a_offsets,
                        "b__values": b_values,
                        "b__offsets": b_offsets,
                    }
                )
            )

    def test_padded_targets(self, sequence_batch, sequence_schema):
        _max_sequence_length = 8
        target_values, target_offsets = _get_values_offsets([[10, 11], [], [12, 13, 14]])
        sequence_batch.targets = {
            "target_1": torch.Tensor([3, 4, 6]),
            "target_2__values": target_values,
            "target_2__offsets": target_offsets,
        }
        padding_op = TabularBatchPadding(
            schema=sequence_schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = padding_op(sequence_batch)

        assert padded_batch.targets["target_2"].shape[1] == _max_sequence_length
        assert torch.equal(padded_batch.targets["target_1"], sequence_batch.targets["target_1"])


class TestTabularSequenceTransform:
    @pytest.fixture
    def sequence_batch(self):
        a_values, a_offsets = _get_values_offsets(data=[[1, 2, 3], [3, 6], [3, 4, 5, 6]])
        b_values, b_offsets = _get_values_offsets([[34, 30, 31], [30, 31], [33, 23, 50, 51]])
        features = {
            "a__values": a_values,
            "a__offsets": a_offsets,
            "b__values": b_values,
            "b__offsets": b_offsets,
            "c_dense": torch.Tensor([[1, 2, 3, 0], [5, 6, 0, 0], [4, 5, 6, 7]]),
            "d_context": torch.Tensor([1, 2, 3, 4]),
        }
        targets = None
        return Batch(features, targets)

    @pytest.fixture
    def sequence_schema(self):
        return Schema(
            [
                ColumnSchema("a", tags=[Tags.SEQUENCE]),
                ColumnSchema("b", tags=[Tags.SEQUENCE]),
                ColumnSchema("c_dense", tags=[Tags.SEQUENCE]),
                ColumnSchema("d_context", tags=[Tags.CONTEXT]),
            ]
        )

    @pytest.fixture
    def padded_batch(self, sequence_schema, sequence_batch):
        _max_sequence_length = 5
        padding_op = TabularBatchPadding(
            schema=sequence_schema, max_sequence_length=_max_sequence_length
        )
        return padding_op(sequence_batch)

    def test_tabular_sequence_transform_wrong_inputs(self, padded_batch, sequence_schema):
        transform = TabularSequenceTransform(
            schema=sequence_schema.select_by_tag(Tags.SEQUENCE), target="a"
        )
        with pytest.raises(
            ValueError,
            match="The input `batch` should include information about input sequences lengths",
        ):
            transform._check_input_sequence_lengths(Batch(padded_batch.features["b"]))

        with pytest.raises(
            ValueError,
            match="Inputs features do not contain target column",
        ):
            transform._check_target_shape(Batch(padded_batch.features["b"]))

        with pytest.raises(
            ValueError, match="must be greater than 1 for sequential input to be shifted as target"
        ):
            transform._check_target_shape(
                Batch(
                    {"a": torch.Tensor([[1, 2], [1, 0], [3, 4]])},
                    sequences=Sequence(lengths={"a": torch.Tensor([2, 1, 2])}),
                )
            )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Sequential target column (d_context) must be a 2D tensor, but shape is 1"
            ),
        ):
            transform = TabularSequenceTransform(schema=sequence_schema, target="d_context")
            transform._check_target_shape(padded_batch)

    def test_transform_predict_next(self, padded_batch, sequence_schema):
        transform = TabularPredictNext(
            schema=sequence_schema.select_by_tag(Tags.SEQUENCE), target="a"
        )
        assert transform.target_name == ["a"]

        batch_output = transform(padded_batch)

        assert list(batch_output.features.keys()) == ["a", "b", "c_dense", "d_context"]
        for k in ["a", "b", "c_dense"]:
            assert torch.equal(batch_output.features[k], padded_batch.features[k][:, :-1])
        assert torch.equal(batch_output.features["d_context"], padded_batch.features["d_context"])
        assert torch.equal(batch_output.sequences.length("a"), torch.Tensor([2, 1, 3]))

    def test_transform_mask_random(self, padded_batch, sequence_schema):
        transform = TabularMaskRandom(
            schema=sequence_schema.select_by_tag(Tags.SEQUENCE), target="a"
        )
        assert transform.target_name == ["a"]

        batch_output = transform(padded_batch)

        assert list(batch_output.features.keys()) == ["a", "b", "c_dense", "d_context"]
        for name in ["a", "b", "c_dense", "d_context"]:
            assert torch.equal(batch_output.features[name], padded_batch.features[name])
        assert torch.equal(batch_output.sequences.length("a"), torch.Tensor([3, 2, 4]))

        # check not all candidates are masked
        pad_mask = padded_batch.features["a"] != 0
        assert torch.all(batch_output.sequences.mask("a").sum(1) != pad_mask.sum(1))
        # check that at least one candidate is masked
        assert torch.all(batch_output.sequences.mask("a").sum(1) > 0)
