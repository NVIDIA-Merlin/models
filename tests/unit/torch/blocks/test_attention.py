import pytest
import torch
from torch import nn

from merlin.models.torch.blocks.attention import CrossAttentionBlock
from merlin.models.torch.utils import module_utils


class TestCrossAttentionBlock:
    def setup_method(self):
        # Set up a simple CrossAttentionBlock instance for testing.
        self.cross = CrossAttentionBlock(
            nn.TransformerEncoderLayer(10, 2, dim_feedforward=10, batch_first=True, dropout=0.0),
            attention=nn.MultiheadAttention(10, 2, batch_first=True),
            key="context",
            seq_key="sequence",
        )
        self.input_dict = {"context": torch.randn(1, 2, 10), "sequence": torch.randn(1, 6, 10)}

    def test_init(self):
        assert self.cross.key == "context"
        assert self.cross.seq_key == "sequence"
        assert isinstance(self.cross.cross_attention, nn.ModuleList)
        assert isinstance(self.cross.cross_attention[0], nn.MultiheadAttention)

    def test_forward(self):
        out = self.cross(self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape == self.input_dict["sequence"].shape

    def test_forward_torch_script(self):
        out = module_utils.module_test(self.cross, self.input_dict)
        assert isinstance(out, torch.Tensor)
        assert out.shape == self.input_dict["sequence"].shape

    def test_get_seq_error(self):
        with pytest.raises(RuntimeError, match="Could not find"):
            self.cross.get_seq(
                {"context": torch.randn(1, 10), "0": torch.randn(1, 10), "1": torch.randn(1, 10)}
            )

        with pytest.raises(
            RuntimeError, match="Please set seq_key for when more than 2 keys are present"
        ):
            cross = CrossAttentionBlock(
                attention=nn.MultiheadAttention(10, 2, batch_first=True),
            )
            cross.get_seq(
                {"context": torch.randn(1, 10), "0": torch.randn(1, 10), "1": torch.randn(1, 10)}
            )
