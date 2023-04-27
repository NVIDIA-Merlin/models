# import pytest
# import torch

# from merlin.io import Dataset
# from merlin.models.torch.blocks.mlp import MLPBlock
# from merlin.models.torch.data import initialize
# from merlin.models.torch.inputs.base import TabularInputBlock
# from merlin.models.torch.inputs.encoder import Encoder
# from merlin.schema import Schema, Tags


# class TestEncoder:
#     def test_init_schema(self, item_id_col_schema, user_id_col_schema):
#         schema = Schema([item_id_col_schema, user_id_col_schema])

#         encoder = Encoder(schema, MLPBlock([10]))
#         assert isinstance(encoder[0], TabularInputBlock)
#         assert encoder.input_schema == schema

#     def test_init_input_block(self, item_id_col_schema, user_id_col_schema):
#         schema = Schema([item_id_col_schema, user_id_col_schema])

#         encoder = Encoder(TabularInputBlock(schema), MLPBlock([10]))
#         assert isinstance(encoder[0], TabularInputBlock)
#         assert encoder.input_schema == schema

#     def test_init_invalid_input(self):
#         with pytest.raises(ValueError):
#             Encoder(MLPBlock([10]))

#     def test_call(self, item_id_col_schema, user_id_col_schema):
#         schema = Schema([item_id_col_schema, user_id_col_schema])

#         encoder = Encoder(schema, MLPBlock([10]))

#         data = {"item_id": torch.tensor([1, 2, 3]), "user_id": torch.tensor([4, 5, 6])}

#         output = encoder(data)

#         assert output.shape == (3, 10)
#         assert torch.equal(encoder.output_shape, torch.tensor([3, 10]))

#         assert encoder.encoder_output_schema.first.dtype.name == "float32"
#         assert encoder.encoder_output_schema.first.shape.as_tuple == ((0, None), 10)
#         assert encoder.encoder_output_schema.first.name == "encoder"

#     def test_prediction(self, music_streaming_data):
#         schema: Schema = music_streaming_data.schema.without(["user_genres", "like", "item_genres"])
#         music_streaming_data.schema = schema

#         encoder = Encoder(schema, MLPBlock([10]))
#         initialize(encoder, music_streaming_data)

#         index = schema.select_by_tag(Tags.USER_ID)

#         predictions = encoder.batch_predict(music_streaming_data, 10, index=index)
#         pred_ddf = predictions.compute(scheduler="synchronous")
#         assert isinstance(predictions, Dataset)
#         assert len(pred_ddf) == 100
#         assert len(pred_ddf.columns) == 22
#         assert pred_ddf.index.name == "user_id"

#         embeddings = encoder.encode(music_streaming_data, 10, index=index)
#         emb_ddf = embeddings.compute(scheduler="synchronous")
#         assert isinstance(embeddings, Dataset)
#         assert len(emb_ddf) == 100
#         assert len(emb_ddf.columns) == 10
#         assert emb_ddf.index.name == "user_id"
