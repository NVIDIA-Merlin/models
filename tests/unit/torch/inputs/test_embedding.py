import numpy as np
import pytest
import torch
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.inputs.embedding import EmbeddingTable, EmbeddingTables
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestEmbeddingTable:
    def test_init(self, item_id_col_schema, user_id_col_schema):
        table = EmbeddingTable(dim=10)
        assert table.dim == 10
        assert not table

        table = EmbeddingTable(lambda col: 10, item_id_col_schema)
        assert table
        assert table.input_schema == Schema([item_id_col_schema])
        assert table.num_embeddings == 11

        table = EmbeddingTable(10, Schema([item_id_col_schema, user_id_col_schema]))
        assert table.num_embeddings == 31
        assert table.extra_repr() == "features: item_id, user_id"
        assert table.contains_multiple_domains()
        assert table.table_name() == "item_id_user_id"

    def test_init_defaults(self, item_id_col_schema):
        table = EmbeddingTable()
        table.initialize_from_schema(item_id_col_schema)

        assert table.dim == 8

    def test_selectable(self, item_id_col_schema, user_id_col_schema):
        table = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))

        user_table = table.select(Tags.USER)
        assert table == user_table

        with pytest.raises(ValueError):
            table.select(None)

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

    def test_forward_tensor(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        input_tensor = torch.tensor([0, 1, 2])
        output = et(input_tensor)

        assert output.size() == (3, 8)
        assert isinstance(output, torch.Tensor)

    def test_forward_dict_single(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema)
        input_dict = {"item_id": torch.tensor([0, 1, 2])}
        output = module_utils.module_test(et, input_dict)

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert output["item_id"].size() == (3, 8)

    def test_multiple_features(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))
        input_dict = {"item_id": torch.tensor([0, 1, 2]), "user_id": torch.tensor([0, 1, 2])}
        output = module_utils.module_test(et, input_dict, batch=mm.Batch(input_dict))

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["item_id"].size() == (3, 8)
        assert output["user_id"].size() == (3, 8)

    def test_multiple_features_different_shapes(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(
            8, Schema([item_id_col_schema, user_id_col_schema]), seq_combiner="mean"
        )
        input_dict = {
            "user_id": torch.tensor([0, 1, 2]),
            "item_id": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        }
        output = module_utils.module_test(et, input_dict, batch=mm.Batch(input_dict))

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["user_id"].size() == (3, 8)
        assert output["item_id"].size() == (3, 8)

    def test_multiple_features_different_shapes_comb(self, item_id_col_schema, user_id_col_schema):
        class Mean(nn.Module):
            def forward(self, x):
                return x.mean(dim=1)

        et = EmbeddingTable(
            8, Schema([item_id_col_schema, user_id_col_schema]), seq_combiner=Mean()
        )
        input_dict = {
            "user_id": torch.tensor([0, 1, 2]),
            "item_id": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        }
        output = module_utils.module_test(et, input_dict, batch=mm.Batch(input_dict))

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["user_id"].size() == (3, 8)
        assert output["item_id"].size() == (3, 8)

    def test_multiple_features_different_shapes_sparse(
        self, item_id_col_schema, user_id_col_schema
    ):
        et = EmbeddingTable(
            8, Schema([item_id_col_schema, user_id_col_schema]), seq_combiner="mean"
        )
        input_dict = {
            "user_id": torch.tensor([0, 1, 2]),
            "item_id__values": torch.tensor([0, 1, 1, 2, 2, 3]),
            "item_id__offsets": torch.tensor([0, 2, 4, 6]),
        }
        output = module_utils.module_test(et, input_dict, batch=mm.Batch(input_dict))

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["user_id"].size() == (3, 8)
        assert output["item_id"].size() == (3, 8)

    def test_multiple_features_sparse(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(
            8, Schema([item_id_col_schema, user_id_col_schema]), seq_combiner="mean"
        )
        input_dict = {
            "user_id__values": torch.tensor([0, 2, 3, 4, 2, 3]),
            "user_id__offsets": torch.tensor([0, 2, 4, 6]),
            "item_id__values": torch.tensor([0, 1, 1, 2, 2, 3]),
            "item_id__offsets": torch.tensor([0, 2, 4, 6]),
        }
        output = module_utils.module_test(et, input_dict, batch=mm.Batch(input_dict))

        assert isinstance(output, dict)
        assert isinstance(output["item_id"], torch.Tensor)
        assert isinstance(output["user_id"], torch.Tensor)
        assert output["user_id"].size() == (3, 8)
        assert output["item_id"].size() == (3, 8)

        with pytest.raises(NotImplementedError):
            et = EmbeddingTable(8, Schema([item_id_col_schema, user_id_col_schema]))
            et(input_dict)

    def test_combiner_module(self, item_id_col_schema):
        et = EmbeddingTable(8, item_id_col_schema, seq_combiner=nn.Linear(8, 10))

        outputs = module_utils.module_test(et, {"item_id": torch.tensor([[0], [1], [2]])})
        assert outputs["item_id"].shape == (3, 1, 10)

    def test_update_feature(self, item_id_col_schema, user_id_col_schema):
        et = EmbeddingTable(8, Schema([user_id_col_schema, item_id_col_schema]))

        assert et.num_embeddings == 31

        updated_user_id = user_id_col_schema.with_properties(
            {"domain": {"min": 0, "max": 30, "name": "user_id"}}
        )
        et.update_feature(updated_user_id)

        with pytest.raises(ValueError):
            et.update_feature(ColumnSchema("test", dtype=np.int32))

        with pytest.raises(ValueError):
            et.update_feature(ColumnSchema("user_id", dtype=np.int32))

        with pytest.raises(ValueError):
            et.update_feature(
                user_id_col_schema.with_properties(
                    {"domain": {"min": 0, "max": 5, "name": "user_id"}}
                )
            )

        assert et.num_embeddings == 41

    def test_exceptions(self, item_id_col_schema):
        table = EmbeddingTable()

        with pytest.raises(RuntimeError):
            table(torch.tensor([0, 1, 2]))

        table.initialize_from_schema(item_id_col_schema)

        with pytest.raises(ValueError):
            table("a")


class TestEmbeddingTables:
    def test_init(self, item_id_col_schema, user_id_col_schema):
        embs = EmbeddingTables(8, Schema([item_id_col_schema, user_id_col_schema]))

        assert len(embs) == 2
        assert "item_id" in embs
        assert "user_id" in embs
        assert embs["item_id"][0].num_embeddings == 11
        assert embs["item_id"][0].dim == 8
        assert embs["user_id"][0].num_embeddings == 21
        assert embs["user_id"][0].dim == 8
        assert embs.extra_repr() == "item_id, user_id"

    def test_select(self, item_id_col_schema, user_id_col_schema):
        embs = EmbeddingTables(8, Schema([item_id_col_schema, user_id_col_schema]))

        user_embs = embs.select(Tags.USER)
        assert isinstance(user_embs, EmbeddingTables)
        assert len(user_embs) == 1
        assert user_embs["user_id"][0].num_embeddings == 21

        item_embs = embs.select(Tags.ITEM)
        assert isinstance(item_embs, EmbeddingTables)
        assert len(item_embs) == 1
        assert item_embs["item_id"][0].num_embeddings == 11

        with pytest.raises(ValueError):
            embs.select(None)

    @pytest.mark.parametrize("nested", [False, True])
    def test_forward(self, item_id_col_schema, user_id_col_schema, nested):
        embs = EmbeddingTables(
            {"user_id": 8, "item_id": 10}, Schema([item_id_col_schema, user_id_col_schema])
        )

        input_dict = {
            "user_id": torch.tensor([0, 1, 2]),
            "item_id": torch.tensor([0, 1, 2]),
        }

        to_call = embs if not nested else mm.ParallelBlock({"a": embs})
        output = module_utils.module_test(to_call, input_dict, batch=mm.Batch(input_dict))

        assert output["item_id"].shape == (3, 10)
        assert output["user_id"].shape == (3, 8)
