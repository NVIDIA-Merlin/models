import pytest
import torch
import torch.nn as nn

from merlin.models.torch.combinators import (
    NoOp,
    ParallelBlock,
    ResidualBlock,
    SequentialBlock,
    SumResidual,
    WithShortcut,
)


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


class TestWithShortcut:
    def test_with_shortcut_init(self):
        linear = nn.Linear(5, 3)
        block = WithShortcut(linear)

        assert isinstance(block, ParallelBlock)
        assert block.parallel_dict["output"] == linear
        assert isinstance(block.parallel_dict["shortcut"], NoOp)
        assert block.post is None
        assert block.aggregation is None

    def test_with_shortcut_forward(self):
        linear = nn.Linear(5, 3)
        block = WithShortcut(linear)

        input_tensor = torch.rand(5, 5)
        output = block(input_tensor)

        assert "output" in output
        assert "shortcut" in output
        assert torch.allclose(output["output"], linear(input_tensor))
        assert torch.allclose(output["shortcut"], input_tensor)

    def test_with_shortcut_post_aggregation(self):
        class PostModule(nn.Module):
            def forward(self, tensors):
                return {k: v + 1 for k, v in tensors.items()}

        class AggregationModule(nn.Module):
            def forward(self, tensors):
                return sum(tensors.values())

        linear = nn.Linear(5, 5)
        post = PostModule()
        aggregation = AggregationModule()
        block = WithShortcut(linear, post=post, aggregation=aggregation)

        input_tensor = torch.rand(5, 5)
        output = block(input_tensor)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 5)

        linear_output = linear(input_tensor)
        expected_output = linear_output + 1 + input_tensor + 1
        assert torch.allclose(output, expected_output)


class TestResidualBlock:
    def test_residual_block_init(self):
        linear = nn.Linear(5, 3)
        block = ResidualBlock(linear)

        assert isinstance(block, WithShortcut)
        assert block.parallel_dict["output"] == linear
        assert block.aggregation.activation is None

        block_with_activation = ResidualBlock(linear, activation="relu")
        assert isinstance(block_with_activation.aggregation, SumResidual)
        assert block_with_activation.aggregation.activation == torch.relu

    def test_residual_block_forward(self):
        linear = nn.Linear(5, 5)
        block = ResidualBlock(linear)

        input_tensor = torch.rand(5, 5)
        output = block(input_tensor)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 5)

        linear_output = linear(input_tensor)
        expected_output = linear_output + input_tensor
        assert torch.allclose(output, expected_output)

    def test_residual_block_forward_with_activation(self):
        linear = nn.Linear(5, 5)
        block = ResidualBlock(linear, activation=nn.ReLU())

        input_tensor = torch.rand(5, 5)
        output = block(input_tensor)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 5)

        linear_output = linear(input_tensor)
        expected_output = nn.ReLU()(linear_output + input_tensor)
        assert torch.allclose(output, expected_output)

    def test_residual_block_shape_mismatch_exception(self):
        linear = nn.Linear(5, 3)
        block = ResidualBlock(linear)

        input_tensor = torch.rand(5, 5)
        with pytest.raises(RuntimeError, match="must have the same shape"):
            block(input_tensor)


class TestSequentialBlock:
    def test_init(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu)

        assert isinstance(block, nn.Sequential)
        assert block[0] == linear
        assert block[1] == relu

    def test_forward(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu)

        input_tensor = torch.rand(5, 5)
        output = block(input_tensor)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 5)

        expected_output = relu(linear(input_tensor))
        assert torch.allclose(output, expected_output)

    def test_append_with_shortcut(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu).append_with_shortcut(linear)

        assert isinstance(block[2], WithShortcut)
        assert block[2]._modules["output"] == linear

    def test_append_with_residual(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu).append_with_residual(linear, activation=nn.ReLU())

        assert isinstance(block[2], ResidualBlock)
        assert block[2]._modules["output"] == linear
        assert isinstance(block[2].aggregation.activation, nn.ReLU)

    def test_append_branch(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu).append_branch(linear, relu)

        assert isinstance(block[2], ParallelBlock)
        assert block[2]._modules["0"] == linear
        assert block[2]._modules["1"] == relu

    def test_repeat(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu).repeat(2)

        assert len(block) == 6
        assert isinstance(block, SequentialBlock)

        assert isinstance(block[2], nn.Linear)
        assert isinstance(block[3], nn.ReLU)
        assert isinstance(block[4], nn.Linear)
        assert isinstance(block[5], nn.ReLU)

        # Check that the weights and biases of the linear layers are not shared
        assert not torch.allclose(block[0].weight, block[2].weight)
        assert not torch.allclose(block[0].bias, block[2].bias)
        assert not torch.allclose(block[0].weight, block[4].weight)
        assert not torch.allclose(block[0].bias, block[4].bias)
        assert not torch.allclose(block[2].weight, block[4].weight)
        assert not torch.allclose(block[2].bias, block[4].bias)

        # Modify the original module weights and biases to ensure the copied
        # modules' weights and biases are not affected
        block[0].weight.data.fill_(1.0)
        block[0].bias.data.fill_(1.0)
        assert not torch.allclose(block[0].weight, block[2].weight)
        assert not torch.allclose(block[0].bias, block[2].bias)
        assert not torch.allclose(block[0].weight, block[4].weight)
        assert not torch.allclose(block[0].bias, block[4].bias)

    def test_repeat_in_parallel(self):
        linear = nn.Linear(5, 5)
        relu = nn.ReLU()
        block = SequentialBlock(linear, relu).repeat_in_parallel(2, shortcut=True)

        assert isinstance(block, ParallelBlock)
        assert len(block) == 3
        assert isinstance(block["0"], SequentialBlock)
        assert isinstance(block["1"], SequentialBlock)
        assert isinstance(block["shortcut"], NoOp)

        # Check that the weights and biases of the linear layers are not shared
        assert not torch.allclose(block["0"][0].weight, block["1"][0].weight)
        assert not torch.allclose(block["0"][0].bias, block["1"][0].bias)

        # Modify the original module weights and biases to ensure the copied
        # modules' weights and biases are not affected
        block["0"][0].weight.data.fill_(1.0)
        block["0"][0].bias.data.fill_(1.0)
        assert not torch.allclose(block["0"][0].weight, block["1"][0].weight)
        assert not torch.allclose(block["0"][0].bias, block["1"][0].bias)
