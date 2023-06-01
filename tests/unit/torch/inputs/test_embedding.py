import numpy as np
import pytest
import torch

from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.schema import ColumnSchema, Schema, Tags


class TestEmbeddingTable:
    def test_init(self, item_id_col_schema, user_id_col_schema):
        table = EmbeddingTable(dim=10)
        assert table.dim == 10
        assert not table

        table = EmbeddingTable(lambda col: 10, item_id_col_schema)
        assert table
        assert table.schema == Schema([item_id_col_schema])
        assert table.num_embeddings == 10

        table = EmbeddingTable(10, Schema([item_id_col_schema, user_id_col_schema]))
        assert table.num_embeddings == 30

    def test_selectable(self, item_id_col_schema, user_id_col_schema):
        table = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))

        user_table = table.select(Tags.USER)
        assert table == user_table

        with pytest.raises(ValueError):
            table.select("unknown")

        with pytest.raises(ValueError):
            table.select(Tags.SEQUENCE)

    def test_add_feature(self, item_id_col_schema, user_id_col_schema):
        sample_column_schema2 = ColumnSchema(
            "item_id2",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
            tags=[Tags.CATEGORICAL],
        )

        etb = EmbeddingTable(8, item_id_col_schema)

        with pytest.raises(ValueError):
            etb.add_feature(ColumnSchema("item_id3", dtype=np.int32))

        # Same domain
        etb.add_feature(sample_column_schema2)
        assert etb.feature_to_domain["item_id2"].name == item_id_col_schema.int_domain.name
        assert etb.num_embeddings == 11

        # Different domain
        etb.add_feature(user_id_col_schema)
        assert etb.feature_to_domain["user_id"].name == user_id_col_schema.int_domain.name

        num_emb = item_id_col_schema.int_domain.max + user_id_col_schema.int_domain.max + 1
        assert etb.num_embeddings == num_emb

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

    def test_multiple_features_different_shapes(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))
        input_dict = {
            "item_id": torch.tensor([0, 1, 2]),
            "user_id": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        }
        output = et(input_dict)

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["item_id"].size() == (3, 8)
        assert output["user_id"].size() == (3, 8)
