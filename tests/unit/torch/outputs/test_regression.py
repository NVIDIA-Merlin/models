#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

import merlin.dtypes as md
import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestRegressionOutput:
    def test_init(self):
        reg_output = mm.RegressionOutput()

        assert isinstance(reg_output, mm.RegressionOutput)
        assert isinstance(reg_output.loss, nn.MSELoss)
        assert reg_output.metrics == [MeanSquaredError()]

    def test_identity(self):
        reg_output = mm.RegressionOutput()
        inputs = torch.randn(3, 2)

        outputs = module_utils.module_test(reg_output, inputs)

        assert outputs.shape == (3, 1)

    def test_output_schema(self):
        schema = ColumnSchema("foo", dtype=md.int16)
        reg_output = mm.RegressionOutput(schema=schema)

        assert isinstance(reg_output.output_schema, Schema)
        assert reg_output.output_schema.first.name == "foo"
        assert reg_output.output_schema.first.dtype == md.float32
        assert Tags.CONTINUOUS in reg_output.output_schema.first.tags

    def test_error_multiple_columns(self):
        with pytest.raises(ValueError, match="Schema must contain exactly one column"):
            mm.RegressionOutput(schema=Schema(["a", "b"]))

    def test_default_loss(self):
        reg_output = mm.RegressionOutput()
        features = torch.randn(3, 2)
        targets = torch.randn(3, 1)

        outputs = module_utils.module_test(reg_output, features)

        assert torch.allclose(
            reg_output.loss(outputs, targets),
            nn.MSELoss()(outputs, targets),
        )

    def test_custom_loss(self):
        reg_output = mm.RegressionOutput(loss=nn.L1Loss())
        features = torch.randn(3, 2)
        targets = torch.randn(3, 1)

        outputs = module_utils.module_test(reg_output, features)

        assert torch.allclose(
            reg_output.loss(outputs, targets),
            nn.L1Loss()(outputs, targets),
        )

    def test_default_metrics(self):
        reg_output = mm.RegressionOutput()
        features = torch.randn(3, 2)
        targets = torch.randn(3, 1)

        outputs = module_utils.module_test(reg_output, features)

        assert torch.allclose(
            reg_output.metrics[0](outputs, targets),
            MeanSquaredError()(outputs, targets),
        )

    def test_custom_metrics(self):
        reg_output = mm.RegressionOutput(
            metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
        )
        features = torch.randn(3, 2)
        targets = torch.randn(3, 1)

        outputs = module_utils.module_test(reg_output, features)

        assert torch.allclose(
            reg_output.metrics[0](outputs, targets),
            MeanAbsoluteError()(outputs, targets),
        )
        assert torch.allclose(
            reg_output.metrics[1](outputs, targets),
            MeanAbsolutePercentageError()(outputs, targets),
        )
