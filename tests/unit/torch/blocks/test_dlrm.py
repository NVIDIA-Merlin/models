import math

import pytest
import torch

import merlin.models.torch as mm
from merlin.models.torch.batch import sample_batch
from merlin.models.torch.blocks.dlrm import DLRMInputBlock, DLRMInteraction
from merlin.models.torch.utils import module_utils
from merlin.schema import Tags


class TestDLRMInputBlock:
    def test_routes_and_output_shapes(self, testing_data):
        schema = testing_data.schema
        embedding_dim = 64
        block = DLRMInputBlock(schema, embedding_dim, mm.MLPBlock([embedding_dim]))

        assert isinstance(block["categorical"], mm.EmbeddingTables)
        assert len(block["categorical"]) == len(schema.select_by_tag(Tags.CATEGORICAL))

        assert isinstance(block["continuous"][0], mm.SelectKeys)
        assert isinstance(block["continuous"][1], mm.MLPBlock)

        batch_size = 16
        batch = sample_batch(testing_data, batch_size=batch_size)

        outputs = module_utils.module_test(block, batch)

        for col in schema.select_by_tag(Tags.CATEGORICAL):
            assert outputs[col.name].shape == (batch_size, embedding_dim)
        assert outputs["continuous"].shape == (batch_size, embedding_dim)


class TestDLRMInteraction:
    @pytest.mark.parametrize(
        "batch_size,num_features,dim",
        [(16, 3, 3), (32, 5, 8), (64, 5, 4)],
    )
    def test_output_shape(self, batch_size, num_features, dim):
        module = DLRMInteraction()
        inputs = torch.rand((batch_size, num_features, dim))
        outputs = module_utils.module_test(module, inputs)

        assert outputs.shape == (batch_size, num_features - 1 + math.comb(num_features - 1, 2))


class TestDLRMBlock:
    @pytest.fixture(autouse=True)
    def setup_method(self, testing_data):
        self.schema = testing_data.schema
        self.batch_size = 16
        self.batch = sample_batch(testing_data, batch_size=self.batch_size)

    def test_dlrm_output_shape(self):
        embedding_dim = 64
        block = mm.DLRMBlock(
            self.schema,
            dim=embedding_dim,
            bottom_block=mm.MLPBlock([embedding_dim]),
        )

        outputs = module_utils.module_test(block, self.batch)

        num_features = len(self.schema.select_by_tag(Tags.CATEGORICAL)) + 1
        dot_product_dim = (num_features - 1) * num_features // 2
        assert list(outputs.shape) == [self.batch_size, dot_product_dim + embedding_dim]

    def test_dlrm_with_top_block(self):
        embedding_dim = 32
        top_block_dim = 8
        block = mm.DLRMBlock(
            self.schema,
            dim=embedding_dim,
            bottom_block=mm.MLPBlock([embedding_dim]),
            top_block=mm.MLPBlock([top_block_dim]),
        )

        outputs = module_utils.module_test(block, self.batch)

        assert list(outputs.shape) == [self.batch_size, top_block_dim]

    def test_dlrm_block_no_categorical_features(self):
        schema = self.schema.remove_by_tag(Tags.CATEGORICAL)
        embedding_dim = 32

        with pytest.raises(ValueError, match="not found in"):
            _ = mm.DLRMBlock(
                schema,
                dim=embedding_dim,
                bottom_block=mm.MLPBlock([embedding_dim]),
            )

    def test_dlrm_block_no_continuous_features(self, testing_data):
        schema = testing_data.schema.remove_by_tag(Tags.CONTINUOUS)
        testing_data.schema = schema

        embedding_dim = 32
        block = mm.DLRMBlock(
            schema,
            dim=embedding_dim,
            bottom_block=mm.MLPBlock([embedding_dim]),
        )

        batch_size = 16
        batch = sample_batch(testing_data, batch_size=batch_size)

        outputs = module_utils.module_test(block, batch)

        num_features = len(schema.select_by_tag(Tags.CATEGORICAL))
        dot_product_dim = (num_features - 1) * num_features // 2
        assert list(outputs.shape) == [batch_size, dot_product_dim]
