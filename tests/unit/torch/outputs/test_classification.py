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
from torchmetrics import AUROC, Accuracy, Precision, Recall
from torchmetrics.classification import BinaryF1Score

import merlin.dtypes as md
import merlin.models.torch as mm
from merlin.models.torch.outputs.classification import CategoricalTarget, EmbeddingTablePrediction
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestBinaryOutput:
    def test_init(self):
        binary_output = mm.BinaryOutput()

        assert isinstance(binary_output, mm.BinaryOutput)
        assert isinstance(binary_output.loss, nn.BCEWithLogitsLoss)
        assert binary_output.metrics == [
            Accuracy(task="binary"),
            AUROC(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
        ]
        assert binary_output.output_schema == Schema()

    def test_identity(self):
        binary_output = mm.BinaryOutput()
        inputs = torch.randn(3, 2)

        outputs = module_utils.module_test(binary_output, inputs)

        assert outputs.shape == (3, 1)

    def test_setup_schema(self):
        schema = ColumnSchema("foo")
        binary_output = mm.BinaryOutput(schema=schema)

        assert isinstance(binary_output.output_schema, Schema)
        assert binary_output.output_schema.first.dtype == md.float32
        assert binary_output.output_schema.first.properties["domain"]["name"] == "foo"
        assert binary_output.output_schema.first.properties["domain"]["min"] == 0
        assert binary_output.output_schema.first.properties["domain"]["max"] == 1

        with pytest.raises(ValueError):
            binary_output.setup_schema(Schema(["a", "b"]))

    def test_custom_loss(self):
        binary_output = mm.BinaryOutput(loss=nn.BCELoss())
        features = torch.randn(3, 2)
        targets = torch.randint(2, (3, 1), dtype=torch.float32)

        outputs = module_utils.module_test(binary_output, features)

        assert torch.allclose(
            binary_output.loss(outputs, targets),
            nn.BCELoss()(outputs, targets),
        )

    def test_cutom_metrics(self):
        binary_output = mm.BinaryOutput(metrics=(BinaryF1Score(),))
        features = torch.randn(3, 2)
        targets = torch.randint(2, (3, 1), dtype=torch.float32)

        outputs = module_utils.module_test(binary_output, features)

        assert torch.allclose(
            binary_output.metrics[0](outputs, targets),
            BinaryF1Score()(outputs, targets),
        )


class TestCategoricalOutput:
    def test_init(self):
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        categorical_output = mm.CategoricalOutput(schema)

        assert isinstance(categorical_output, mm.CategoricalOutput)
        assert isinstance(categorical_output.loss, nn.CrossEntropyLoss)
        assert sorted(m.__class__.__name__ for m in categorical_output.metrics) == [
            "MulticlassAveragePrecision",
            "MulticlassPrecision",
            "MulticlassRecall",
        ]
        output_schema = categorical_output.output_schema.first
        assert output_schema.dtype == md.float32
        assert output_schema.properties["domain"]["min"] == 0
        assert output_schema.properties["domain"]["max"] == int_domain_max
        assert (
            output_schema.properties["value_count"]["min"]
            == output_schema.properties["value_count"]["max"]
            == int_domain_max + 1
        )

    def test_called_with_schema(self):
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        categorical_output = mm.CategoricalOutput(schema)

        inputs = torch.randn(3, 2)
        outputs = module_utils.module_test(categorical_output, inputs)

        num_classes = int_domain_max + 1
        assert outputs.shape == (3, num_classes)

    def test_called_with_categorical_target(self):
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        target = mm.CategoricalTarget(schema)
        categorical_output = mm.CategoricalOutput(target)

        inputs = torch.randn(3, 2)
        outputs = module_utils.module_test(categorical_output, inputs)

        num_classes = int_domain_max + 1
        assert outputs.shape == (3, num_classes)

    def test_called_with_embedding_table(self):
        embedding_dim = 8
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        table = mm.EmbeddingTable(embedding_dim, schema)
        categorical_output = mm.CategoricalOutput(table)

        inputs = torch.randn(3, embedding_dim)
        outputs = module_utils.module_test(categorical_output, inputs)

        num_classes = int_domain_max + 1
        assert outputs.shape == (3, num_classes)

    def test_invalid_type_error(self):
        with pytest.raises(ValueError, match="Invalid to_call type"):
            mm.CategoricalOutput("invalid to_call")


class TestCategoricalTarget:
    def test_init(self, user_id_col_schema):
        schema = Schema([user_id_col_schema])

        # Test with ColumnSchema
        model = CategoricalTarget(user_id_col_schema)
        assert model.num_classes == user_id_col_schema.int_domain.max + 1
        assert isinstance(model.linear, nn.LazyLinear)

        # Test with Schema
        model = CategoricalTarget(feature=schema)
        assert model.num_classes == user_id_col_schema.int_domain.max + 1
        assert isinstance(model.linear, nn.LazyLinear)

    def test_forward(self, user_id_col_schema):
        model = CategoricalTarget(feature=user_id_col_schema)

        inputs = torch.randn(5, 11)
        output = model(inputs)

        assert output.shape == (5, 21)

    def test_forward_with_activation(self, user_id_col_schema):
        model = CategoricalTarget(feature=user_id_col_schema, activation=nn.ReLU())

        inputs = torch.randn(5, 11)
        output = model(inputs)

        assert output.shape == (5, 21)
        assert torch.all(output >= 0)

    def test_embedding_lookup(self, user_id_col_schema):
        model = CategoricalTarget(feature=user_id_col_schema)

        model(torch.randn(5, 11))  # initialize the embedding table
        input_indices = torch.tensor([1, 5, 10])
        hidden_vectors = model.embedding_lookup(input_indices)

        assert hidden_vectors.shape == (3, 11)
        assert model.embeddings().shape == (11, 21)


class TestEmbeddingTablePrediction:
    def test_forward(self, user_id_col_schema):
        input_block = mm.TabularInputBlock(
            Schema([user_id_col_schema]), init="defaults", agg="concat"
        )
        user_emb = input_block.select(Tags.USER_ID).leaf()
        prediction = EmbeddingTablePrediction(user_emb)

        inputs = torch.randn(5, 8)
        output = prediction(inputs)

        assert output.shape == (5, 21)

    def test_embedding_lookup(self):
        embedding_dim = 8
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        table = mm.EmbeddingTable(embedding_dim, schema)
        model = mm.EmbeddingTablePrediction(table)

        batch_size = 16
        ids = torch.randint(0, int_domain_max, (batch_size,))
        outputs = model.embedding_lookup(ids)

        assert outputs.shape == (batch_size, embedding_dim)
        assert model.embeddings().shape == (int_domain_max + 1, embedding_dim)
