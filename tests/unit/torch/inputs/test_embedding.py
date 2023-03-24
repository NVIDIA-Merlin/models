import numpy as np
import pytest
import torch

from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.inputs.embedding import Embeddings, EmbeddingTable, EmbeddingTableModule
from merlin.schema import ColumnSchema, Schema, Tags


class TestEmbeddingTableModule:
    def test_init(self, item_id_col_schema):
        with pytest.raises(TypeError):
            _ = EmbeddingTableModule(8)

        etb = EmbeddingTableModule(8, item_id_col_schema)
        assert etb.dim == 8
        assert len(etb.domains) == 1
        assert etb.feature_to_domain["item_id"].name == item_id_col_schema.int_domain.name

    def test_selectable(self, item_id_col_schema, user_id_col_schema):
        etb = EmbeddingTableModule(8, Schema([item_id_col_schema, user_id_col_schema]))

        assert etb.select_by_tag(Tags.ITEM_ID).schema == Schema([item_id_col_schema])
        assert etb.select_by_name("item_id").schema == Schema([item_id_col_schema])
        assert etb.select_by_tag(Tags.USER_ID).schema == Schema([user_id_col_schema])
        assert etb.select_by_name("user_id").schema == Schema([user_id_col_schema])

    def test_add_feature(self, item_id_col_schema, user_id_col_schema):
        sample_column_schema2 = ColumnSchema(
            "item_id2",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
            tags=[Tags.CATEGORICAL],
        )

        etb = EmbeddingTableModule(8, item_id_col_schema)

        with pytest.raises(ValueError):
            etb.add_feature(ColumnSchema("item_id3", dtype=np.int32))

        # Same domain
        etb.add_feature(sample_column_schema2)
        assert etb.feature_to_domain["item_id2"].name == item_id_col_schema.int_domain.name

        # Different domain
        etb.add_feature(user_id_col_schema)
        assert etb.feature_to_domain["user_id"].name == user_id_col_schema.int_domain.name

        num_emb = item_id_col_schema.int_domain.max + user_id_col_schema.int_domain.max + 1
        assert etb.num_embeddings == num_emb


class TestEmbeddingTable:
    def test_init(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        assert et.dim == 8
        assert et.schema["item_id"] == item_id_col_schema
        assert et.table.weight.size() == (11, 8)
        assert et.sequence_combiner is None

    def test_forward(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        input_tensor = torch.tensor([0, 1, 2])
        output = et(input_tensor)

        assert output.size() == (3, 8)
        assert isinstance(output, torch.Tensor)

    def test_forward_dict_single(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        input_dict = {"item_id": torch.tensor([0, 1, 2])}
        output = et(input_dict)

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert output["item_id"].size() == (3, 8)

    def test_multiple_features(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))
        input_dict = {"item_id": torch.tensor([0, 1, 2]), "user_id": torch.tensor([0, 1, 2])}
        output = et(input_dict)

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["item_id"].size() == (3, 8)
        assert output["user_id"].size() == (3, 8)


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
