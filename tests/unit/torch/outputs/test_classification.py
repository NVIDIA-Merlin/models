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
import torchmetrics as tm
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
        assert isinstance(binary_output.loss, nn.BCELoss)
        assert binary_output.metrics == [
            Accuracy(task="binary"),
            AUROC(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
        ]
        with pytest.raises(ValueError):
            mm.output_schema(binary_output)

    def test_identity(self):
        binary_output = mm.BinaryOutput()
        inputs = torch.randn(3, 2)

        outputs = module_utils.module_test(binary_output, inputs)

        assert outputs.shape == (3, 1)

    def test_output_schema(self):
        schema = ColumnSchema("foo")
        binary_output = mm.BinaryOutput(schema=schema)

        assert isinstance(binary_output.output_schema, Schema)
        assert binary_output.output_schema.first.dtype == md.float32
        assert binary_output.output_schema.first.properties["domain"]["name"] == "foo"
        assert binary_output.output_schema.first.properties["domain"]["min"] == 0
        assert binary_output.output_schema.first.properties["domain"]["max"] == 1

    def test_error_multiple_columns(self):
        with pytest.raises(ValueError, match="Schema must contain exactly one column"):
            mm.BinaryOutput(schema=Schema(["a", "b"]))

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
        assert isinstance(categorical_output.metrics[0], tm.RetrievalHitRate)
        assert isinstance(categorical_output.metrics[1], tm.RetrievalNormalizedDCG)
        assert isinstance(categorical_output.metrics[2], tm.RetrievalPrecision)
        assert isinstance(categorical_output.metrics[3], tm.RetrievalRecall)

        output_schema = categorical_output[0].output_schema.first
        assert output_schema.dtype == md.float32
        assert output_schema.properties["domain"]["min"] == 0
        assert output_schema.properties["domain"]["max"] == 1
        assert (
            output_schema.properties["value_count"]["min"]
            == output_schema.properties["value_count"]["max"]
            == int_domain_max + 1
        )
        assert mm.output_schema(categorical_output) == categorical_output[0].output_schema

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

    def test_weight_tying(self):
        embedding_dim = 8
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        table = mm.EmbeddingTable(embedding_dim, schema)
        categorical_output = mm.CategoricalOutput.with_weight_tying(table)

        inputs = torch.randn(3, embedding_dim)
        outputs = module_utils.module_test(categorical_output, inputs)

        num_classes = int_domain_max + 1
        assert outputs.shape == (3, num_classes)

        cat_output = mm.CategoricalOutput(schema).tie_weights(table)
        assert isinstance(cat_output[0], EmbeddingTablePrediction)

    def test_invalid_type_error(self):
        with pytest.raises(ValueError, match="Target must be a ColumnSchema or Schema"):
            mm.CategoricalOutput("invalid to_call")

    def test_multiple_column_schema_error(self, item_id_col_schema, user_id_col_schema):
        schema = Schema([item_id_col_schema])
        assert len(schema) == 1
        _ = mm.CategoricalOutput(schema)

        schema_with_two_columns = schema + Schema([user_id_col_schema])
        assert len(schema_with_two_columns) == 2
        with pytest.raises(ValueError, match="must contain exactly one"):
            _ = mm.CategoricalOutput(schema_with_two_columns)


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

    def test_forward_model_output(self):
        int_domain_max = 3
        schema = (
            ColumnSchema("foo")
            .with_dtype(md.int32)
            .with_properties({"domain": {"name": "bar", "min": 0, "max": int_domain_max}})
        )
        target = mm.CategoricalTarget(schema)
        categorical_output = mm.ModelOutput(target, loss=nn.CrossEntropyLoss())
        assert mm.output_schema(categorical_output).column_names == ["foo"]

        inputs = torch.randn(3, 2)
        outputs = module_utils.module_test(categorical_output, inputs)
        num_classes = int_domain_max + 1
        assert outputs.shape == (3, num_classes)


class TestEmbeddingTablePrediction:
    def test_init_multiple_int_domains(self, user_id_col_schema, item_id_col_schema):
        input_block = mm.TabularInputBlock(Schema([user_id_col_schema, item_id_col_schema]))
        input_block.add_route(Tags.CATEGORICAL, mm.EmbeddingTable(10))
        table = mm.schema.select(input_block, Tags.USER_ID).leaf()

        with pytest.raises(ValueError):
            EmbeddingTablePrediction(table)

        with pytest.raises(ValueError):
            EmbeddingTablePrediction.with_weight_tying(input_block)

        with pytest.raises(ValueError):
            EmbeddingTablePrediction.with_weight_tying(input_block, "a")

        with pytest.raises(ValueError):
            EmbeddingTablePrediction.with_weight_tying(input_block, Tags.CATEGORICAL)

        assert isinstance(EmbeddingTablePrediction(table, Tags.USER_ID), EmbeddingTablePrediction)
        assert isinstance(
            EmbeddingTablePrediction.with_weight_tying(input_block, Tags.USER_ID),
            EmbeddingTablePrediction,
        )

    def test_forward(self, user_id_col_schema):
        input_block = mm.TabularInputBlock(
            Schema([user_id_col_schema]), init="defaults", agg="concat"
        )
        prediction = EmbeddingTablePrediction.with_weight_tying(input_block, Tags.USER_ID)

        inputs = torch.randn(5, 8)
        output = module_utils.module_test(prediction, inputs)

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
