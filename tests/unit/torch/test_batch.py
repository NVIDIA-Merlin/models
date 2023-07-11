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

import pytest
import torch

from merlin.dataloader.torch import Loader
from merlin.models.torch.batch import Batch, Sequence, sample_batch, sample_features


class TestSequence:
    @pytest.fixture
    def sequence(self):
        lengths = {"feature1": torch.tensor([4, 5]), "feature2": torch.tensor([3, 7])}
        masks = {
            "feature1": torch.tensor([[1, 0], [1, 1]]),
            "feature2": torch.tensor([[1, 1], [1, 0]]),
        }
        return Sequence(lengths, masks)

    def test_contains(self, sequence):
        assert "feature1" in sequence
        assert "feature3" not in sequence

    def test_length(self, sequence):
        assert torch.equal(sequence.length("feature1"), torch.tensor([4, 5]))
        with pytest.raises(ValueError):
            sequence.length("feature3")

    def test_mask(self, sequence):
        assert torch.equal(sequence.mask("feature1"), torch.tensor([[1, 0], [1, 1]]))
        with pytest.raises(ValueError):
            sequence.mask("feature3")

    def test_with_incorrect_types(self):
        with pytest.raises(ValueError):
            Sequence("not a tensor or dict", "not a tensor or dict")

    def test_init_tensor_lengths(self):
        # Test when lengths is a tensor
        lengths = torch.tensor([1.0])
        sequence = Sequence(lengths)

        assert isinstance(sequence.lengths, dict)
        assert "default" in sequence.lengths
        assert torch.equal(sequence.lengths["default"], lengths)
        assert sequence.device() == lengths.device

    def test_init_tensor_masks(self):
        # Test when masks is a tensor
        lengths = torch.tensor([1.0])
        masks = torch.tensor([2.0])
        sequence = Sequence(lengths, masks)

        assert isinstance(sequence.masks, dict)
        assert "default" in sequence.masks
        assert torch.equal(sequence.masks["default"], masks)

    def test_init_no_masks(self):
        # Test when masks is None
        lengths = torch.tensor([1.0])
        sequence = Sequence(lengths)

        assert isinstance(sequence.masks, dict)
        assert len(sequence.masks) == 0

    def test_init_invalid_lengths(self):
        # Test when lengths is not a tensor nor a dictionary of tensors
        lengths = "invalid_lengths"

        with pytest.raises(ValueError, match="Lengths must be a tensor or a dictionary of tensors"):
            Sequence(lengths)

    def test_init_invalid_masks(self):
        # Test when masks is not a tensor nor a dictionary of tensors
        lengths = torch.tensor([1.0])
        masks = "invalid_masks"

        with pytest.raises(ValueError, match="Masks must be a tensor or a dictionary of tensors"):
            Sequence(lengths, masks)

    def test_device(self):
        empty_seq = Sequence({})

        with pytest.raises(ValueError, match="Sequence is empty"):
            empty_seq.device()


class TestBatch:
    @pytest.fixture
    def batch(self):
        features = {"feature1": torch.tensor([1, 2]), "feature2": torch.tensor([3, 4])}
        targets = {"target1": torch.tensor([0, 1])}
        lengths = {"feature1": torch.tensor([4, 5]), "feature2": torch.tensor([3, 7])}
        masks = {
            "feature1": torch.tensor([[1, 0], [1, 1]]),
            "feature2": torch.tensor([[1, 1], [1, 0]]),
        }
        sequences = Sequence(lengths, masks)
        return Batch(features, targets, sequences)

    def test_replace(self, batch):
        new_features = {"feature1": torch.tensor([5, 6]), "feature2": torch.tensor([7, 8])}
        new_targets = {"target1": torch.tensor([1, 0])}
        lengths = {"feature1": torch.tensor([6, 7]), "feature2": torch.tensor([8, 9])}
        masks = {
            "feature1": torch.tensor([[1, 1], [1, 0]]),
            "feature2": torch.tensor([[0, 1], [1, 1]]),
        }
        new_sequences = Sequence(lengths, masks)

        new_batch = batch.replace(new_features, new_targets, new_sequences)
        assert torch.equal(new_batch.features["feature1"], new_features["feature1"])
        assert torch.equal(new_batch.targets["target1"], new_targets["target1"])
        assert torch.equal(new_batch.sequences.length("feature1"), new_sequences.length("feature1"))

    def test_batch_init_tensor_target(self):
        # Test when targets is a tensor
        features = torch.tensor([1.0])
        targets = torch.tensor([2.0])
        batch = Batch(features, targets)

        assert isinstance(batch.targets, dict)
        assert "default" in batch.targets
        assert torch.equal(batch.targets["default"], targets)
        assert batch.device() == features.device

    def test_batch_init_invalid_targets(self):
        # Test when targets is not a tensor nor a dictionary of tensors
        features = torch.tensor([1.0])
        targets = "invalid_target"

        with pytest.raises(ValueError, match="Targets must be a tensor or a dictionary of tensors"):
            Batch(features, targets)

    def test_feature(self, batch):
        assert torch.equal(batch.feature("feature1"), torch.tensor([1, 2]))
        with pytest.raises(ValueError):
            batch.feature("feature3")

    def test_target(self, batch):
        assert torch.equal(batch.target("target1"), torch.tensor([0, 1]))
        with pytest.raises(ValueError):
            batch.target("target2")

    def test_bool(self, batch):
        assert bool(batch)
        empty_batch = Batch({}, {})
        assert not bool(empty_batch)

    def test_with_incorrect_types(self):
        with pytest.raises(ValueError):
            Batch("not a tensor or dict", "not a tensor or dict", "not a sequence")

    def test_sample(self, music_streaming_data):
        batch = Batch.sample_from(music_streaming_data)
        assert isinstance(batch, Batch)

        assert isinstance(batch.features, dict)
        assert len(list(batch.features.keys())) == 12
        for key, val in batch.features.items():
            if not key.endswith("__values") and not key.endswith("__offsets"):
                assert val.shape[0] == 32

        assert isinstance(batch.targets, dict)
        assert list(batch.targets.keys()) == ["click", "play_percentage", "like"]
        for val in batch.targets.values():
            assert val.shape[0] == 32

    def test_device(self):
        empty_batch = Batch({}, {})

        with pytest.raises(ValueError, match="Batch is empty"):
            empty_batch.device()

    def test_flatten_as_dict(self):
        features = {"feature1": torch.tensor([1, 2]), "feature2": torch.tensor([3, 4])}
        targets = {"target1": torch.tensor([5, 6])}
        lengths = {"length1": torch.tensor([7, 8])}
        masks = {"mask1": torch.tensor([9, 10])}
        sequences = Sequence(lengths, masks)
        batch = Batch(features, targets, sequences)

        # without inputs
        result = batch.flatten_as_dict(batch)
        assert len(result) == 5  # 2 features, 1 target, 1 length, 1 mask
        assert set(result.keys()) == set(
            [
                "features.feature1",
                "features.feature2",
                "targets.target1",
                "lengths.length1",
                "masks.mask1",
            ]
        )

        # with inputs
        input_batch = Batch(
            {"feature2": torch.tensor([11, 12])}, {"target1": torch.tensor([13, 14])}, sequences
        )
        result = batch.flatten_as_dict(input_batch)
        assert len(result) == 9  # input keys are considered
        assert (
            len([k for k in result if k.startswith("inputs.")]) == 4
        )  # 1 feature, 1 target, 1 length, 1 mask

    def test_from_partial_dict(self):
        features = {"feature1": torch.tensor([1, 2]), "feature2": torch.tensor([3, 4])}
        targets = {"target1": torch.tensor([5, 6])}
        lengths = {"length1": torch.tensor([7, 8])}
        masks = {"mask1": torch.tensor([9, 10])}
        sequences = Sequence(lengths, masks)

        batch = Batch(features, targets, sequences)

        partial_dict = {
            "features.feature1": torch.tensor([11, 12]),
            "targets.target1": torch.tensor([13, 14]),
            "lengths.length1": torch.tensor([15, 16]),
            "inputs.features.feature1": torch.tensor([1]),
            "inputs.targets.target1": torch.tensor([1]),
            "inputs.lengths.length1": torch.tensor([1]),
            "inputs.masks.mask1": torch.tensor([1]),
        }

        # create a batch from partial_dict
        result = Batch.from_partial_dict(partial_dict, batch)

        assert result.features["feature1"].equal(
            torch.tensor([11, 12])
        )  # updated from partial_dict
        assert result.features["feature2"].equal(torch.tensor([3, 4]))  # kept from batch
        assert result.targets["target1"].equal(torch.tensor([13, 14]))  # updated from partial_dict
        assert result.sequences.lengths["length1"].equal(
            torch.tensor([15, 16])
        )  # updated from partial_dict
        assert "mask1" not in result.sequences.masks  # removed from batch


class Test_sample_batch:
    def test_loader(self, music_streaming_data):
        loader = Loader(music_streaming_data, batch_size=2)

        batch = sample_batch(loader)

        assert isinstance(batch.features, dict)
        assert len(list(batch.features.keys())) == 12
        for key, val in batch.features.items():
            if not key.endswith("__values") and not key.endswith("__offsets"):
                assert val.shape[0] == 2

        assert isinstance(batch.targets, dict)
        assert list(batch.targets.keys()) == ["click", "play_percentage", "like"]
        for val in batch.targets.values():
            assert val.shape[0] == 2

    def test_dataset(self, music_streaming_data):
        batch = sample_batch(music_streaming_data, batch_size=2)

        assert isinstance(batch.features, dict)
        assert len(list(batch.features.keys())) == 12
        for key, val in batch.features.items():
            if not key.endswith("__values") and not key.endswith("__offsets"):
                assert val.shape[0] == 2

        assert isinstance(batch.targets, dict)
        assert list(batch.targets.keys()) == ["click", "play_percentage", "like"]
        for val in batch.targets.values():
            assert val.shape[0] == 2

    def test_exceptions(self, music_streaming_data):
        with pytest.raises(ValueError, match="specify 'batch_size'"):
            sample_batch(music_streaming_data)

        with pytest.raises(ValueError, match="Expected Dataset or Loader instance"):
            sample_batch(torch.tensor([1, 2, 3]))


class Test_sample_features:
    def test_no_targets(self, music_streaming_data):
        features = sample_features(music_streaming_data, batch_size=2)

        assert isinstance(features, dict)
        assert len(list(features.keys())) == 12
        for key, val in features.items():
            if not key.endswith("__values") and not key.endswith("__offsets"):
                assert val.shape[0] == 2
