import numpy as np
import pytest
import torch

from merlin.models.torch.core.combinators import ParallelBlock
from merlin.models.torch.inputs.embedding import Embeddings, EmbeddingTable, EmbeddingTableBase
from merlin.schema import ColumnSchema, Schema, Tags


class TestEmbeddingTableBase:
    def test_init(self, item_id_col_schema):
        with pytest.raises(ValueError):
            _ = EmbeddingTableBase(8)

        etb = EmbeddingTableBase(8, item_id_col_schema)
        assert etb.dim == 8
        assert etb.features["item_id"] == item_id_col_schema

    def test_add_feature(self, item_id_col_schema):
        sample_column_schema2 = ColumnSchema(
            "item_id2",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
            tags=[Tags.CATEGORICAL],
        )

        etb = EmbeddingTableBase(8, item_id_col_schema)

        with pytest.raises(ValueError):
            etb.add_feature(ColumnSchema("item_id3", dtype=np.int32))

        with pytest.raises(ValueError):
            etb.add_feature(
                ColumnSchema(
                    "item_id4",
                    dtype=np.int32,
                    properties={"domain": {"min": 0, "max": 15, "name": "item_id"}},
                )
            )

        etb.add_feature(sample_column_schema2)
        assert etb.features["item_id2"] == sample_column_schema2


class TestEmbeddingTable:
    def test_init(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        assert et.dim == 8
        assert et.features["item_id"] == item_id_col_schema
        assert et.table.weight.size() == (11, 8)
        assert et.sequence_combiner is None
        assert et.l2_batch_regularization_factor == 0.0

    def test_forward(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        input_tensor = torch.tensor([0, 1, 2])
        output = et(input_tensor)

        assert output.size() == (3, 8)
        assert isinstance(output, torch.Tensor)


class TestEmbeddings:
    def test_embeddings_init(self, item_id_col_schema, user_id_col_schema):
        schema = Schema([item_id_col_schema, user_id_col_schema])
        embeddings = Embeddings(schema)

        assert isinstance(embeddings, ParallelBlock)
        assert len(embeddings) == 2
        assert isinstance(embeddings._modules["item_id"], EmbeddingTable)
        assert isinstance(embeddings._modules["user_id"], EmbeddingTable)

    def test_embeddings_with_custom_dim(self, item_id_col_schema, user_id_col_schema):
        schema = Schema([item_id_col_schema, user_id_col_schema])
        dim = {"item_id": 8, "user_id": 16}
        embeddings = Embeddings(schema, dim=dim)

        assert isinstance(embeddings, ParallelBlock)
        assert len(embeddings) == 2
        assert isinstance(embeddings["item_id"], EmbeddingTable)
        assert isinstance(embeddings["user_id"], EmbeddingTable)
        assert embeddings["item_id"].dim == 8
        assert embeddings["user_id"].dim == 16
