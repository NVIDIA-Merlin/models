import torch
from torch import nn

from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.data import Batch, Sequence
from merlin.models.torch.utils import module_utils


class TestTabularSequence:
    def test_basic(self):
        lengths = {"feature1": torch.tensor([4, 5]), "feature2": torch.tensor([3, 7])}
        masks = {
            "feature1": torch.tensor([[1, 0], [1, 1]]),
            "feature2": torch.tensor([[1, 1], [1, 0]]),
        }

        tab_seq = Sequence(lengths, masks)

        assert len(tab_seq.lengths) == 2
        assert len(tab_seq.masks) == 2

        assert torch.equal(tab_seq.lengths["feature1"], lengths["feature1"])
        assert torch.equal(tab_seq.lengths["feature2"], lengths["feature2"])

        assert torch.equal(tab_seq.masks["feature1"], masks["feature1"])
        assert torch.equal(tab_seq.masks["feature2"], masks["feature2"])

        assert "feature1" in tab_seq
        assert "feature2" in tab_seq
        assert "feature3" not in tab_seq


class TestTabularBatch:
    def test_basic(self):
        features = {"feature1": torch.tensor([1, 2]), "feature2": torch.tensor([3, 4])}
        targets = {"target1": torch.tensor([0, 1])}

        lengths = {"feature1": torch.tensor([4, 5]), "feature2": torch.tensor([3, 7])}
        masks = {
            "feature1": torch.tensor([[1, 0], [1, 1]]),
            "feature2": torch.tensor([[1, 1], [1, 0]]),
        }
        sequences = Sequence(lengths, masks)

        tab_batch = Batch(features, targets, sequences)

        assert len(tab_batch.features) == 2
        assert len(tab_batch.targets) == 1
        assert tab_batch.sequences is not None

        assert torch.equal(tab_batch.features["feature1"], features["feature1"])
        assert torch.equal(tab_batch.features["feature2"], features["feature2"])

        assert torch.equal(tab_batch.targets["target1"], targets["target1"])

        new_features = {"feature1": torch.tensor([5, 6]), "feature2": torch.tensor([7, 8])}
        new_targets = {"target1": torch.tensor([1, 0])}

        new_tab_batch = tab_batch.replace(features=new_features, targets=new_targets)

        assert torch.equal(new_tab_batch.features["feature1"], new_features["feature1"])
        assert torch.equal(new_tab_batch.features["feature2"], new_features["feature2"])

        assert torch.equal(new_tab_batch.targets["target1"], new_targets["target1"])

        assert new_tab_batch.sequences is not None
        assert bool(new_tab_batch)

        empty_tab_batch = Batch({}, {})
        assert not bool(empty_tab_batch)
