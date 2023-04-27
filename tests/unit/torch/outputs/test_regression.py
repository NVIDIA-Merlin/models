# import pytest
# import torch
# from torch import nn
# from torchmetrics import MeanSquaredError

# from merlin.models.torch.blocks.mlp import MLPBlock
# from merlin.models.torch.combinators import SequentialBlock
# from merlin.models.torch.outputs.base import ModelOutput
# from merlin.models.torch.outputs.regression import RegressionOutput
# from merlin.schema import ColumnSchema


# class TestRegressionOutput:
#     def test_inheritance(self):
#         ro = RegressionOutput()
#         assert isinstance(ro, ModelOutput)

#     def test_initialization(self):
#         ro = RegressionOutput()
#         assert isinstance(ro.to_call, (nn.LazyLinear, nn.Linear))
#         assert isinstance(ro.default_loss, nn.MSELoss)
#         assert isinstance(ro.default_metrics[0], MeanSquaredError)
#         assert ro.target is None
#         assert ro.pre is None
#         assert ro.post is None

#     def test_with_target_str(self):
#         ro = RegressionOutput(target="target_column")
#         assert ro.target == "target_column"

#     def test_with_target_column_schema(self):
#         cs = ColumnSchema("target_column")
#         ro = RegressionOutput(target=cs)
#         assert ro.target == "target_column"

#     def test_not_implemented_temperature(self):
#         with pytest.raises(NotImplementedError):
#             RegressionOutput(logits_temperature=0.3)

#     def test_custom_to_call(self):
#         custom_to_call = nn.Linear(10, 1)
#         ro = RegressionOutput(to_call=custom_to_call)
#         assert isinstance(ro.to_call, nn.Linear)

#     def test_forward(self):
#         batch = torch.randn(2, 3)

#         ro = RegressionOutput()
#         output = ro(batch)

#         assert output.shape == (2, 1)

#     def test_forward_with_mlp(self):
#         batch = torch.randn(2, 3)

#         block = SequentialBlock(MLPBlock([5, 3]), RegressionOutput())
#         output = block(batch)

#         assert output.shape == (2, 1)
