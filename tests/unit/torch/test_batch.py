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

from merlin.models.torch.batch import Batch, Sequence


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
