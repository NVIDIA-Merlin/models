from itertools import accumulate

import pytest
import torch

from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.transforms.sequences import (
    BroadcastToSequence,
    TabularPadding,
    TabularPredictNext,
)
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


def _get_values_offsets(data):
    values = []
    row_lengths = []
    for row in data:
        row_lengths.append(len(row))
        values += row
    offsets = [0] + list(accumulate(row_lengths))
    return torch.tensor(values), torch.tensor(offsets)


class TestTabularPadding:
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
        padding_op = TabularPadding(
            schema=sequence_schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = module_utils.module_test(padding_op, sequence_batch)

        assert torch.equal(padded_batch.sequences.length("a"), torch.Tensor([2, 0, 3]))
        assert set(padded_batch.features.keys()) == set(["a", "b", "c_dense", "d_context"])
        for feature in ["a", "b", "c_dense"]:
            assert padded_batch.features[feature].shape[1] == _max_sequence_length

    def test_batch_invalid_lengths(self):
        # Test when targets is not a tensor nor a dictionary of tensors
        a_values, a_offsets = _get_values_offsets(data=[[1, 2], [], [3, 4, 5]])
        b_values, b_offsets = _get_values_offsets([[34], [23, 56], [33, 23, 50, 4]])

        with pytest.raises(
            ValueError,
            match="The sequential inputs must have the same length for each row in the batch",
        ):
            padding_op = TabularPadding(schema=Schema(["a", "b"]), selection=None)
            padding_op(
                inputs=None,
                batch=Batch(
                    {
                        "a__values": a_values,
                        "a__offsets": a_offsets,
                        "b__values": b_values,
                        "b__offsets": b_offsets,
                    }
                ),
            )

    def test_padded_targets(self, sequence_batch, sequence_schema):
        _max_sequence_length = 8
        target_values, target_offsets = _get_values_offsets([[10, 11], [], [12, 13, 14]])
        sequence_batch.targets = {
            "target_1": torch.Tensor([3, 4, 6]),
            "target_2__values": target_values,
            "target_2__offsets": target_offsets,
        }
        padding_op = TabularPadding(
            schema=sequence_schema, max_sequence_length=_max_sequence_length
        )
        padded_batch = module_utils.module_test(padding_op, sequence_batch)

        assert padded_batch.targets["target_2"].shape[1] == _max_sequence_length
        assert torch.equal(padded_batch.targets["target_1"], sequence_batch.targets["target_1"])


class TestBroadcastToSequence:
    def setup_method(self):
        self.input_tensors = {
            "feature_1": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "feature_2": torch.tensor(
                [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]]
            ),
            "feature_3": torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        }
        self.schema = Schema(list(self.input_tensors.keys()))
        self.to_broadcast = Schema(["feature_1", "feature_3"])
        self.sequence = Schema(["feature_2"])
        self.broadcast = BroadcastToSequence(self.to_broadcast, self.sequence)

    def test_initialize_from_schema(self):
        self.broadcast.initialize_from_schema(self.schema)
        assert self.broadcast.to_broadcast_features == ["feature_1", "feature_3"]
        assert self.broadcast.sequence_features == ["feature_2"]

    def test_get_seq_length(self):
        self.broadcast.initialize_from_schema(self.schema)
        assert self.broadcast.get_seq_length(self.input_tensors) == 3

    def test_get_seq_length_offsets(self):
        self.broadcast.initialize_from_schema(self.schema)

        inputs = {
            "feature_1": torch.tensor([1, 2]),
            "feature_2__offsets": torch.tensor([2, 3]),
            "feature_3": torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        }

        assert self.broadcast.get_seq_length(inputs) == 3

    def test_forward(self):
        self.broadcast.initialize_from_schema(self.schema)
        output = module_utils.module_test(self.broadcast, self.input_tensors)
        assert output["feature_1"].shape == (2, 3, 3)
        assert output["feature_3"].shape == (2, 3, 3)
        assert output["feature_2"].shape == (2, 3, 2)

    def test_unsupported_dimensions(self):
        self.broadcast.initialize_from_schema(self.schema)
        self.input_tensors["feature_3"] = torch.rand(10, 3, 3, 3)

        with pytest.raises(RuntimeError, match="Unsupported number of dimensions: 4"):
            self.broadcast(self.input_tensors)


class TestTabularPredictNext:
    @pytest.fixture
    def sequence_batch(self):
        a_values, a_offsets = _get_values_offsets(data=[[1, 2, 3], [3, 6], [3, 4, 5, 6]])
        b_values, b_offsets = _get_values_offsets([[34, 30, 31], [30, 31], [33, 23, 50, 51]])

        c_values, c_offsets = _get_values_offsets([[1, 2, 3, 4], [5, 6], [5, 6, 7, 8, 9, 10]])
        d_values, d_offsets = _get_values_offsets(
            [[10, 20, 30, 40], [50, 60], [50, 60, 70, 80, 90, 100]]
        )

        features = {
            "a__values": a_values,
            "a__offsets": a_offsets,
            "b__values": b_values,
            "b__offsets": b_offsets,
            "c__values": c_values,
            "c__offsets": c_offsets,
            "d__values": d_values,
            "d__offsets": d_offsets,
            "e_dense": torch.Tensor([[1, 2, 3, 0], [5, 6, 0, 0], [4, 5, 6, 7]]),
            "f_context": torch.Tensor([1, 2, 3, 4]),
        }
        targets = None
        return Batch(features, targets)

    @pytest.fixture
    def sequence_schema_1(self):
        return Schema(
            [
                ColumnSchema("a", tags=[Tags.SEQUENCE]),
                ColumnSchema("b", tags=[Tags.SEQUENCE]),
                ColumnSchema("e_dense", tags=[Tags.SEQUENCE]),
            ]
        )

    @pytest.fixture
    def sequence_schema_2(self):
        return Schema(
            [
                ColumnSchema("c", tags=[Tags.SEQUENCE, Tags.ID]),
                ColumnSchema("d", tags=[Tags.SEQUENCE]),
            ]
        )

    @pytest.fixture
    def padded_batch(self, sequence_schema_1, sequence_batch):
        padding_op = TabularPadding(schema=sequence_schema_1)
        return padding_op(sequence_batch)

    def test_tabular_sequence_transform_wrong_inputs(self, padded_batch, sequence_schema_1):
        with pytest.raises(
            ValueError,
            match="The target 'Tags.ID' was not found in the provided sequential schema:",
        ):
            transform = TabularPredictNext(
                schema=sequence_schema_1,
                target=Tags.ID,
            )

        transform = TabularPredictNext(
            schema=sequence_schema_1,
            target="a",
            apply_padding=False,
        )
        with pytest.raises(
            ValueError,
            match="The input `batch` should include information about input sequences lengths",
        ):
            transform(Batch({"b": padded_batch.features["b"]}))

        with pytest.raises(
            ValueError,
            match="Inputs features do not contain target column",
        ):
            transform(Batch({"b": padded_batch.features["b"]}, sequences=padded_batch.sequences))

        with pytest.raises(
            ValueError, match="must be greater than 1 for sequential input to be shifted as target"
        ):
            transform = TabularPredictNext(
                schema=sequence_schema_1.select_by_name("a"), target="a", apply_padding=False
            )
            transform(
                Batch(
                    {"a": torch.Tensor([[1, 2], [1, 0], [3, 4]])},
                    sequences=Sequence(lengths={"a": torch.Tensor([2, 1, 2])}),
                )
            )

    def test_transform_predict_next(self, sequence_batch, padded_batch, sequence_schema_1):
        transform = TabularPredictNext(schema=sequence_schema_1, target="a")

        batch_output = module_utils.module_test(transform, sequence_batch)

        assert list(batch_output.features.keys()) == ["a", "b", "e_dense"]
        for k in ["a", "b", "e_dense"]:
            assert torch.equal(batch_output.features[k], padded_batch.features[k][:, :-1])
        assert torch.equal(batch_output.sequences.length("a"), torch.Tensor([2, 1, 3]))

    def test_transform_predict_next_multi_sequence(
        self, sequence_batch, padded_batch, sequence_schema_1, sequence_schema_2
    ):
        import merlin.models.torch as mm

        transform_1 = TabularPredictNext(schema=sequence_schema_1, target="a")
        transform_2 = TabularPredictNext(schema=sequence_schema_2)
        transform_block = mm.BatchBlock(
            mm.ParallelBlock({"transform_1": transform_1, "transform_2": transform_2})
        )
        batch_output = transform_block(sequence_batch)

        assert list(batch_output.features.keys()) == ["a", "b", "e_dense", "f_context", "c", "d"]
        assert list(batch_output.targets.keys()) == ["a", "c"]

        assert torch.equal(batch_output.sequences.length("a"), torch.Tensor([2, 1, 3]))
        assert torch.equal(batch_output.sequences.length("c"), torch.Tensor([3, 1, 5]))
        assert torch.all(
            batch_output.sequences.mask("a")  # target mask
            == torch.Tensor(
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, True],
                ]
            )
        )
