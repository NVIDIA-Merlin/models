import pytest
import torch

import merlin.models.torch as mm
from merlin.models.torch.inputs.embedding import infer_embedding_dim
from merlin.models.torch.utils.selection_utils import select_schema
from merlin.schema import Schema, Tags


class TestTabularInputBlock:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema
        self.input_block: mm.TabularInputBlock = mm.TabularInputBlock(self.schema)
        self.batch: mm.Batch = mm.Batch.sample_from(music_streaming_data, batch_size=10)

    def test_forward(self):
        self.input_block.add_route(Tags.CONTINUOUS)
        self.input_block.add_route(Tags.CATEGORICAL, mm.EmbeddingTable(10, seq_combiner="mean"))

        outputs = self.input_block(self.batch.features)

        for name in select_schema(self.schema, Tags.CONTINUOUS).column_names:
            assert name in outputs

        for name in select_schema(self.schema, Tags.CATEGORICAL).column_names:
            assert name in outputs
            assert outputs[name].shape == (10, 10)

    def test_init_detaults(self):
        input_block = mm.TabularInputBlock(self.schema, init="defaults")
        outputs = input_block(self.batch.features)

        for name in select_schema(self.schema, Tags.CONTINUOUS).column_names:
            assert name in outputs

        for name in select_schema(self.schema, Tags.CATEGORICAL).column_names:
            assert name in outputs
            assert outputs[name].shape == (
                10,
                infer_embedding_dim(self.schema.select_by_name(name)),
            )

    def test_init_agg(self):
        input_block = mm.TabularInputBlock(self.schema, init="defaults", agg="concat")
        outputs = input_block(self.batch.features)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (10, 107)

    def test_exceptions(self):
        with pytest.raises(ValueError):
            mm.TabularInputBlock(self.schema, init="unknown")
