import pytest
import torch

import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema, Tags


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

        outputs = module_utils.module_test(output_block, torch.rand(10, 10))

        assert "play_percentage" in outputs
        assert "click" in outputs
        assert "like" in outputs

    def test_exceptions(self):
        with pytest.raises(ValueError, match="not found"):
            mm.TabularOutputBlock(self.schema, init="not_found")

        outputs = mm.TabularOutputBlock(self.schema)
        with pytest.raises(ValueError):
            outputs.add_route(Tags.CATEGORICAL)
