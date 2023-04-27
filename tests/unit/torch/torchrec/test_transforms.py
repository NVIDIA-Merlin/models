# from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# from merlin.models.torch.data import sample_batch
# from merlin.models.torch.torchrec.transforms import ToKeyedJaggedTensor


# class TestToKeyedJaggedTensor:
#     def test_forward(self, music_streaming_data):
#         batch = sample_batch(
#             music_streaming_data, batch_size=10, shuffle=False, include_targets=False
#         )

#         convertor = ToKeyedJaggedTensor(music_streaming_data.schema)

#         keyed_jagged = convertor(batch)

#         assert isinstance(keyed_jagged, KeyedJaggedTensor)
