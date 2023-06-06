import pytest
import torch
from torch import nn

from merlin.models.torch.block import Block
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.utils import module_utils


class TestMLPBlock:
    def test_init(self):
        units = (32, 64, 128)
        mlp = MLPBlock(units)
        assert isinstance(mlp, MLPBlock)
        assert isinstance(mlp, Block)
        assert len(mlp) == len(units) * 2 + 1

    def test_activation(self):
        units = [32, 64, 128]
        mlp = MLPBlock(units, activation=nn.ReLU)
        assert isinstance(mlp, MLPBlock)
        for i, module in enumerate(mlp[1:]):
            if i % 2 == 1:
                assert isinstance(module, nn.ReLU)

    def test_normalization_batch_norm(self):
        units = [32, 64, 128]
        mlp = MLPBlock(units, normalization="batchnorm")
        assert isinstance(mlp, MLPBlock)
        for i, module in enumerate(mlp[1:]):
            if (i + 1) % 3 == 0:
                assert isinstance(module, nn.LazyBatchNorm1d)

    def test_normalization_custom(self):
        units = [32, 64, 128]
        custom_norm = nn.LayerNorm(1)
        mlp = MLPBlock(units, normalization=custom_norm)
        assert isinstance(mlp, MLPBlock)
        for i, module in enumerate(mlp[1:]):
            if i % 3 == 2:
                assert isinstance(module, nn.LayerNorm)

    def test_normalization_invalid(self):
        units = [32, 64, 128]
        with pytest.raises(ValueError):
            MLPBlock(units, normalization="invalid")

    def test_dropout_float(self):
        units = [32, 64, 128]
        mlp = MLPBlock(units, dropout=0.5)
        assert isinstance(mlp, MLPBlock)
        for i, module in enumerate(mlp[1:]):
            if i % 3 == 2:
                assert isinstance(module, nn.Dropout)
                assert module.p == 0.5

    def test_dropout_module(self):
        units = [32, 64, 128]
        mlp = MLPBlock(units, dropout=nn.Dropout(0.5))
        assert isinstance(mlp, MLPBlock)
        for i, module in enumerate(mlp[1:]):
            if i % 3 == 2:
                assert isinstance(module, nn.Dropout)
                assert module.p == 0.5

    def test_forward(self):
        mlp = MLPBlock([32])
        inputs = {"a": torch.randn(32, 2), "b": torch.randn(32, 2)}
        outputs = module_utils.module_test(mlp, inputs)
        assert outputs.shape == torch.Size([32, 32])
