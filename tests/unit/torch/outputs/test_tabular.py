import pytest
import torch

import merlin.models.torch as mm
from merlin.schema import Schema


class TestTabularOutputBlock:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema
        self.batch: mm.Batch = mm.Batch.sample_from(music_streaming_data, batch_size=10)

    def test_init_defaults(self):
        output_block = mm.TabularOutputBlock(self.schema, init="defaults")

        assert isinstance(output_block["play_percentage"], mm.RegressionOutput)
        assert isinstance(output_block["click"], mm.BinaryOutput)
        assert isinstance(output_block["like"], mm.BinaryOutput)

        outputs = output_block(torch.rand(10, 10))

        assert "play_percentage" in outputs
        assert "click" in outputs
        assert "like" in outputs
