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
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, Precision, Recall

import merlin.dtypes as md
import merlin.models.torch as mm
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema


class TestBinaryOutput:
    def test_init(self):
        binary_output = mm.BinaryOutput()

        assert isinstance(binary_output, mm.BinaryOutput)
        assert isinstance(binary_output.loss, nn.BCEWithLogitsLoss)
        assert binary_output.metrics == (
            Accuracy(task="binary"),
            AUROC(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
        )
        assert binary_output.output_schema is None

    def test_identity(self):
        binary_output = mm.BinaryOutput()
        inputs = torch.randn(2, 3)

        outputs = module_utils.module_test(binary_output, inputs)

        assert outputs.shape == (2, 1)

    def test_setup_schema(self):
        schema = ColumnSchema("foo")
        binary_output = mm.BinaryOutput(schema=schema)

        assert isinstance(binary_output.output_schema, Schema)
        assert binary_output.output_schema.first.dtype == md.float32
        assert binary_output.output_schema.first.properties["domain"]["name"] == "foo"
        assert binary_output.output_schema.first.properties["domain"]["min"] == 0
        assert binary_output.output_schema.first.properties["domain"]["max"] == 1
