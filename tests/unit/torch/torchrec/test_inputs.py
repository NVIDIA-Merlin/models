# import itertools

# import pytest
# import torch
# from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
# from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedTensor

# from merlin.models.torch.data import get_device, sample_batch
# from merlin.models.torch.torchrec.inputs import (
#     EmbeddingBagCollection,
#     EmbeddingCollection,
#     FusedEmbeddingBagCollection,
#     FusedEmbeddingCollection,
#     create_embedding_configs,
# )
# from merlin.schema import Schema, Tags


# class Test_create_embedding_configs:
#     def test_table_configs_fixed_dim(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)
#         configs = create_embedding_configs(schema, dim=10)

#         assert len(schema) == len(configs) + 1  # There is one shared table
#         assert all(c.embedding_dim == 10 for c in configs)

#         by_name = {c.name: c for c in configs}
#         assert set(by_name["genres"].feature_names) == {"user_genres", "item_genres"}
#         assert by_name["genres"].num_embeddings == 101

#     def test_table_configs_infer_dim(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)
#         configs = create_embedding_configs(schema)

#         assert len(schema) == len(configs) + 1  # There is one shared table

#         by_name = {c.name: c for c in configs}
#         assert set(by_name["genres"].feature_names) == {"user_genres", "item_genres"}
#         assert by_name["genres"].num_embeddings == 101

#         assert by_name["genres"].embedding_dim == 8
#         assert by_name["item_category"].embedding_dim == 8
#         assert by_name["country"].embedding_dim == 8
#         assert by_name["session_id"].embedding_dim == 24
#         assert by_name["item_id"].embedding_dim == 24
#         assert by_name["user_id"].embedding_dim == 24

#     def test_table_configs_with_kwargs(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)

#         data_types = {
#             "genres": DataType.INT2,
#             "item_category": DataType.INT4,
#             "country": DataType.INT8,
#             "session_id": DataType.UINT8,
#             "item_id": DataType.FP16,
#             "user_id": DataType.FP32,
#         }

#         configs = create_embedding_configs(
#             schema, config_cls=EmbeddingBagConfig, pooling=PoolingType.SUM, data_type=data_types
#         )

#         assert len(schema) == len(configs) + 1  # There is one shared table
#         assert all(c.pooling == PoolingType.SUM for c in configs)
#         assert all(c.data_type == data_types[c.name] for c in configs)


# class TestEmbeddingBagCollection:
#     def test_init(self, user_id_col_schema, item_id_col_schema):
#         schema = Schema([user_id_col_schema, item_id_col_schema])
#         embs = EmbeddingBagCollection(schema)

#         assert embs.input_schema == schema
#         assert set(itertools.chain(*embs._feature_names)) == set(schema.column_names)

#     @pytest.mark.parametrize("post", [None, "to-dict"])
#     def test_forward(self, user_id_col_schema, item_id_col_schema, post):
#         embs = EmbeddingBagCollection(
#             Schema([user_id_col_schema, item_id_col_schema]), dim=4, post=post
#         )

#         data = {
#             "user_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#             "item_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#         }
#         outputs = embs(data)

#         assert outputs["user_id"].shape == (10, 4)
#         assert outputs["item_id"].shape == (10, 4)
#         if post:
#             assert isinstance(outputs, dict)
#         else:
#             assert isinstance(outputs, KeyedTensor)

#     def test_forward_music_streaming_data(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_name(["user_genres", "item_genres"])
#         embs = EmbeddingBagCollection(schema, dim=4)

#         data = sample_batch(
#             music_streaming_data, batch_size=10, shuffle=False, include_targets=False
#         )
#         embs.to(get_device(data))

#         outputs = embs(data)

#         assert set(outputs.keys()) == set(schema.column_names)
#         assert all(outputs[k].shape == (10, 4) for k in outputs.to_dict())


# class TestEmbeddingCollection:
#     def test_init(self, user_id_col_schema, item_id_col_schema):
#         schema = Schema([user_id_col_schema, item_id_col_schema])
#         embs = EmbeddingCollection(schema)

#         assert embs.input_schema == schema
#         assert set(itertools.chain(*embs._feature_names)) == set(schema.column_names)

#     def test_forward_music_streaming_data(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_name(["user_genres", "item_genres"])
#         embs = EmbeddingCollection(schema, dim=4)

#         data = sample_batch(
#             music_streaming_data, batch_size=10, shuffle=False, include_targets=False
#         )
#         embs.to(get_device(data))

#         outputs = embs(data)

#         assert set(outputs.keys()) == set(schema.column_names)
#         assert all(isinstance(outputs[k], JaggedTensor) for k in outputs)


# class TestFusedEmbeddingBagCollection:
#     def test_init(self, user_id_col_schema, item_id_col_schema):
#         schema = Schema([user_id_col_schema, item_id_col_schema])
#         embs = FusedEmbeddingBagCollection(
#             schema,
#             optimizer_type=torch.optim.SGD,
#             optimizer_kwargs={"lr": 0.02},
#         )

#         assert embs.input_schema == schema
#         assert set(embs.embedding_bags.keys()) == set(schema.column_names)

#     @pytest.mark.parametrize("post", [None, "to-dict"])
#     def test_forward(self, user_id_col_schema, item_id_col_schema, post):
#         embs = FusedEmbeddingBagCollection(
#             Schema([user_id_col_schema, item_id_col_schema]),
#             optimizer_type=torch.optim.SGD,
#             optimizer_kwargs={"lr": 0.02},
#             dim=4,
#             post=post,
#         )

#         data = {
#             "user_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#             "item_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#         }
#         outputs = embs(data)

#         assert outputs["user_id"].shape == (10, 4)
#         assert outputs["item_id"].shape == (10, 4)
#         if post:
#             assert isinstance(outputs, dict)
#         else:
#             assert isinstance(outputs, KeyedTensor)

#     def test_forward_music_streaming_data(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_name(["user_genres", "item_genres"])
#         data = sample_batch(
#             music_streaming_data, batch_size=10, shuffle=False, include_targets=False
#         )
#         embs = FusedEmbeddingBagCollection(
#             schema,
#             dim=4,
#             optimizer_type=torch.optim.SGD,
#             optimizer_kwargs={"lr": 0.02},
#             device=get_device(data),
#         )

#         outputs = embs(data)

#         assert set(outputs.keys()) == set(schema.column_names)
#         assert all(outputs[k].shape == (10, 4) for k in outputs.to_dict())


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# class TestFusedEmbeddingCollection:
#     def test_init(self, user_id_col_schema, item_id_col_schema):
#         schema = Schema([user_id_col_schema, item_id_col_schema])
#         embs = FusedEmbeddingCollection(
#             schema,
#             optimizer_type=torch.optim.SGD,
#             optimizer_kwargs={"lr": 0.02},
#             device=torch.device("cuda"),
#         )

#         assert embs.input_schema == schema
#         assert set(c.name for c in embs._embedding_configs) == set(schema.column_names)

#     def test_forward_music_streaming_data(self, music_streaming_data):
#         schema = music_streaming_data.schema.select_by_name(["user_genres", "item_genres"])

#         data = sample_batch(
#             music_streaming_data, batch_size=10, shuffle=False, include_targets=False
#         )
#         embs = FusedEmbeddingCollection(
#             schema,
#             dim=4,
#             optimizer_type=torch.optim.SGD,
#             optimizer_kwargs={"lr": 0.02},
#             device=get_device(data),
#         )
#         outputs = embs(data)

#         assert set(outputs.keys()) == set(schema.column_names)
#         assert all(isinstance(outputs[k], JaggedTensor) for k in outputs)
