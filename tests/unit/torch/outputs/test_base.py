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
import numpy as np
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy

import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestModelOutput:
    def test_init(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        model_output = mm.ModelOutput(block, loss=loss)

        assert isinstance(model_output, mm.ModelOutput)
        assert model_output.loss is loss
        assert model_output.metrics == ()
        assert model_output.output_schema == Schema()

    def test_identity(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        model_output = mm.ModelOutput(block, loss=loss)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(model_output, inputs)

        assert torch.equal(inputs, outputs)

    def test_setup_metrics(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        metrics = (Accuracy(task="binary"), AUROC(task="binary"))
        model_output = mm.ModelOutput(block, loss=loss, metrics=metrics)

        assert model_output.metrics == metrics

    def test_setup_schema(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        schema = ColumnSchema("feature", dtype=np.int32, tags=[Tags.CONTINUOUS])
        model_output = mm.ModelOutput(block, loss=loss, schema=schema)

        assert isinstance(model_output.output_schema, Schema)
        assert model_output.output_schema.first == schema

    def test_eval_resets_target(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        model_output = mm.ModelOutput(block, loss=loss)

        assert torch.equal(model_output.target, torch.zeros(1))
        model_output.target = torch.ones(1)
        assert torch.equal(model_output.target, torch.ones(1))
        model_output.eval()
        assert torch.equal(model_output.target, torch.zeros(1))
