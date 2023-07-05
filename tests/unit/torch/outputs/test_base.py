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
from torchmetrics import AUROC, Accuracy

import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils


class TestModelOutput:
    def test_init(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        model_output = mm.ModelOutput(block, loss=loss)

        assert isinstance(model_output, mm.ModelOutput)
        assert model_output.loss is loss
        assert model_output.metrics is None
        assert not mm.schema.output_schema(model_output)

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
        metrics = [Accuracy(task="binary"), AUROC(task="binary")]
        model_output = mm.ModelOutput(block, loss=loss, metrics=metrics)

        assert model_output.metrics == metrics

    def test_eval_resets_target(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        model_output = mm.ModelOutput(block, loss=loss)

        assert torch.equal(model_output.target, torch.zeros(1))
        model_output.target = torch.ones(1)
        assert torch.equal(model_output.target, torch.ones(1))
        model_output.eval()
        assert torch.equal(model_output.target, torch.zeros(1))

    def test_copy(self):
        block = mm.Block()
        loss = nn.BCEWithLogitsLoss()
        metrics = [Accuracy(task="multiclass", num_classes=11)]
        model_output = mm.ModelOutput(block, loss=loss, metrics=metrics)

        model_copy = model_output.copy()
        assert model_copy.loss is not loss
        assert isinstance(model_copy.loss, nn.BCEWithLogitsLoss)
        assert model_copy.metrics[0] is not metrics[0]
        assert model_copy.metrics[0].__class__.__name__ == "MulticlassAccuracy"
        assert model_copy.metrics[0].num_classes == 11

    @pytest.mark.parametrize("logits_temperature", [0.1, 0.9])
    def test_logits_temperature_scaler(self, logits_temperature):
        block = mm.Block()
        model_output = mm.ModelOutput(block, logits_temperature=logits_temperature)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(model_output, inputs)

        assert torch.allclose(inputs / logits_temperature, outputs)
