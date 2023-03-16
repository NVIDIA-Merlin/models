import pytest
import torch
import torch.nn as nn

from merlin.models.torch.core.combinators import ParallelBlock


class TestParallelBlock:
    def test_single_branch(self):
        linear = nn.Linear(2, 3)
        parallel_block = ParallelBlock(linear)
        x = torch.randn(4, 2)
        out = parallel_block(x)

        assert isinstance(out, dict)
        assert len(out) == 1
        assert torch.allclose(out[0], linear(x))

    def test_branch_list(self):
        layers = [nn.Linear(2, 3), nn.ReLU(), nn.Linear(2, 1)]
        parallel_block = ParallelBlock(*layers)
        x = torch.randn(4, 2)
        out = parallel_block(x)

        assert isinstance(out, dict)
        assert len(out) == len(layers)
        assert set(out.keys()) == {0, 1, 2}

        for i, layer in enumerate(layers):
            assert torch.allclose(out[i], layer(x))

    def test_branch_dict(self):
        layers_dict = {"linear": nn.Linear(2, 3), "relu": nn.ReLU(), "linear2": nn.Linear(2, 1)}
        parallel_block = ParallelBlock(layers_dict)
        x = torch.randn(4, 2)
        out = parallel_block(x)

        assert isinstance(out, dict)
        assert len(out) == len(layers_dict)
        assert list(out.keys()) == list(layers_dict.keys())

        for name, layer in layers_dict.items():
            assert torch.allclose(out[name], layer(x))

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _ = ParallelBlock("invalid_input")

    def test_pre_post_aggregation(self):
        class PreModule(nn.Module):
            def forward(self, inputs):
                return inputs * 2

        class PostModule(nn.Module):
            def forward(self, inputs):
                return {"post": inputs[0]}

        class AggModule(nn.Module):
            def forward(self, inputs):
                return inputs["post"]

        layers = [nn.Linear(2, 3), nn.ReLU(), nn.Linear(2, 1)]
        parallel_block = ParallelBlock(
            *layers, pre=PreModule(), post=PostModule(), aggregation=AggModule()
        )
        x = torch.randn(4, 2)
        out = parallel_block(x)

        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, layers[0](PreModule()(x)))
