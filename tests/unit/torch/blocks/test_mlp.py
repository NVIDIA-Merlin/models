# import pytest
# from torch import nn

# from merlin.models.torch.blocks.mlp import MLPBlock
# from merlin.models.torch.combinators import SequentialBlock


# class TestMLPBlock:
#     def test_init(self):
#         units = [32, 64, 128]
#         mlp = MLPBlock(units)
#         assert isinstance(mlp, MLPBlock)
#         assert isinstance(mlp, SequentialBlock)
#         assert len(mlp) == len(units) * 2

#     def test_activation(self):
#         units = [32, 64, 128]
#         mlp = MLPBlock(units, activation=nn.ReLU)
#         assert isinstance(mlp, MLPBlock)
#         for i, module in enumerate(mlp):
#             if i % 2 == 1:
#                 assert isinstance(module, nn.ReLU)

#     def test_normalization_batch_norm(self):
#         units = [32, 64, 128]
#         mlp = MLPBlock(units, normalization="batch_norm")
#         assert isinstance(mlp, MLPBlock)
#         for i, module in enumerate(mlp):
#             if (i + 1) % 3 == 0:
#                 assert isinstance(module, nn.LazyBatchNorm1d)

#     def test_normalization_custom(self):
#         units = [32, 64, 128]
#         custom_norm = nn.LayerNorm(1)
#         mlp = MLPBlock(units, normalization=custom_norm)
#         assert isinstance(mlp, MLPBlock)
#         for i, module in enumerate(mlp):
#             if i % 3 == 2:
#                 assert isinstance(module, nn.LayerNorm)

#     def test_normalization_invalid(self):
#         units = [32, 64, 128]
#         with pytest.raises(ValueError):
#             MLPBlock(units, normalization="invalid")

#     def test_dropout_float(self):
#         units = [32, 64, 128]
#         mlp = MLPBlock(units, dropout=0.5)
#         assert isinstance(mlp, MLPBlock)
#         for i, module in enumerate(mlp):
#             if i % 3 == 2:
#                 assert isinstance(module, nn.Dropout)
#                 assert module.p == 0.5

#     def test_dropout_module(self):
#         units = [32, 64, 128]
#         mlp = MLPBlock(units, dropout=nn.Dropout(0.5))
#         assert isinstance(mlp, MLPBlock)
#         for i, module in enumerate(mlp):
#             if i % 3 == 2:
#                 assert isinstance(module, nn.Dropout)
#                 assert module.p == 0.5

#     def test_pre_post(self):
#         units = [32, 64, 128]
#         pre = nn.Linear(1, 1)
#         post = nn.Linear(1, 1)
#         mlp = MLPBlock(units, pre=pre, post=post)
#         assert isinstance(mlp, MLPBlock)
#         assert mlp.pre == pre
#         assert mlp.post == post
