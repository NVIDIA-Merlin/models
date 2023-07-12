import pytest
import torch

import merlin.dtypes as md
import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


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

    def test_init_defaults_with_binary_categorical(self):
        test_schema = Schema(
            [
                ColumnSchema("foo")
                .with_dtype(md.int32)
                .with_properties({"domain": {"name": "bar", "min": 0, "max": 1}})
                .with_tags([Tags.CATEGORICAL, Tags.TARGET])
            ]
        )
        output_block = mm.TabularOutputBlock(test_schema, init="defaults")

        assert isinstance(output_block["foo"], mm.BinaryOutput)

        outputs = module_utils.module_test(output_block, torch.rand(10, 10))

        assert "foo" in outputs

    def test_init_defaults_with_multiclass_categorical(self):
        test_schema = Schema(
            [
                ColumnSchema("foo")
                .with_dtype(md.int32)
                .with_properties({"domain": {"name": "bar", "min": 0, "max": 3}})
                .with_tags([Tags.CATEGORICAL, Tags.TARGET])
            ]
        )
        output_block = mm.TabularOutputBlock(test_schema, init="defaults")

        assert isinstance(output_block["foo"], mm.CategoricalOutput)

        outputs = module_utils.module_test(output_block, torch.rand(10, 10))

        assert "foo" in outputs

    def test_exceptions(self):
        with pytest.raises(ValueError, match="not found"):
            mm.TabularOutputBlock(self.schema, init="not_found")

    def test_no_route_for_non_existent_tag(self):
        outputs = mm.TabularOutputBlock(self.schema)
        outputs.add_route(Tags.CATEGORICAL, required=False)

        assert not outputs

    def test_nesting(self):
        output_block = mm.TabularOutputBlock(self.schema)
        output_block.add_route(Tags.TARGET, mm.TabularOutputBlock(init="defaults"))

        outputs = module_utils.module_test(output_block, torch.rand(10, 10))

        assert "play_percentage" in outputs
        assert "click" in outputs
        assert "like" in outputs
