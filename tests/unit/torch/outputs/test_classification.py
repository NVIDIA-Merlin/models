import numpy as np
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, Precision, Recall

from merlin.models.torch.inputs.embedding import Embeddings
from merlin.models.torch.outputs.classification import (
    BinaryOutput,
    CategoricalOutput,
    CategoricalTarget,
)
from merlin.schema import ColumnSchema, Schema, Tags

binary_column = ColumnSchema(
    "binary_column",
    dtype=np.int32,
    properties={"domain": {"min": 0, "max": 1, "name": "binary_column"}},
    tags=[Tags.CATEGORICAL, Tags.TARGET],
)

multiclass_column = ColumnSchema(
    "multiclass_column",
    dtype=np.int32,
    properties={"domain": {"min": 0, "max": 15, "name": "multiclass_column"}},
    tags=[Tags.CATEGORICAL, Tags.TARGET],
)


class TestBinaryOutput:
    def test_column_schema(self):
        model = BinaryOutput(binary_column)
        model.default_metrics = (
            Accuracy("binary"),
            AUROC("binary"),
            Precision("binary"),
            Recall("binary"),
        )

        inputs = torch.randn(5, 11)
        output = model(inputs)

        assert output.shape == (5, 1)


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
        model = CategoricalTarget(feature=user_id_col_schema, activation=nn.ReLU)

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


class TestCategoricalOutput:
    def test_column_schema(self):
        model = CategoricalOutput(multiclass_column)

        inputs = torch.randn(5, 11)
        output = model(inputs)

        assert output.shape == (5, multiclass_column.int_domain.max + 1)

    def test_embedding_table(self, music_streaming_data):
        schema: Schema = music_streaming_data.schema.without(["user_genres", "item_genres"])
        music_streaming_data.schema = schema

        input_block = Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=8)
        output_block = CategoricalOutput(input_block.select_by_tag(Tags.ITEM_ID).first)

        inputs = torch.randn(5, 8)
        output = output_block(inputs)

        dim = schema.select_by_tag(Tags.ITEM_ID).first.int_domain.max + 1
        assert output.shape == (5, dim)
