import pytest

from merlin.models.utils.schema_utils import (
    get_embedding_size_from_cardinality,
    select_schema,
    selection_name,
)
from merlin.schema import ColumnSchema, Schema, Tags


class TestSelectSchema:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema

    def test_select_schema(self):
        selection = self.schema.select_by_tag(Tags.USER)
        output = select_schema(self.schema, selection)

        assert output == selection

    def test_select_tag(self):
        selection = self.schema.select_by_tag(Tags.USER)
        output = select_schema(self.schema, Tags.USER)

        assert output == selection

    def test_select_callable(self):
        def selection_callable(schema: Schema):
            return schema.select_by_tag(Tags.USER)

        selection = self.schema.select_by_tag(Tags.USER)
        output = select_schema(self.schema, selection_callable)
        assert output == selection

    def test_select_column(self):
        column = self.schema["user_id"]

        output = select_schema(self.schema, column)
        output_2 = select_schema(self.schema, ColumnSchema("user_id"))
        assert output == column == output_2

    def test_exceptions(self):
        with pytest.raises(ValueError, match="is not valid"):
            select_schema(self.schema, 1)

        with pytest.raises(ValueError, match="is not valid"):
            select_schema(1, 1)


class TestSelectionName:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema

    def test_select_schema(self):
        selection = self.schema.select_by_tag(Tags.USER)

        assert selection_name(selection) == "_".join(selection.column_names)

    def test_select_tag(self):
        selection = Tags.USER

        assert selection_name(selection) == selection.value

    def test_select_callable(self):
        def selection_callable(schema: Schema):
            return schema.select_by_tag(Tags.USER)

        assert selection_name(selection_callable) == selection_callable.__name__

    def test_select_column(self):
        column = self.schema["user_id"]

        assert selection_name(column) == column.name
        assert selection_name(ColumnSchema("user_id")) == column.name

    def test_exceptions(self):
        with pytest.raises(ValueError, match="is not valid"):
            selection_name(1)


@pytest.mark.parametrize(
    "cardinality_x_expected_dim",
    [
        (1, 5),
        (5, 8),
        (10, 9),
        (20, 11),
        (50, 14),
        (100, 16),
        (500, 24),
        (1000, 29),
        (5000, 43),
        (10000, 50),
        (50000, 75),
        (100000, 89),
        (500000, 133),
        (1000000, 159),
        (5000000, 237),
        (10000000, 282),
    ],
)
def test_get_embedding_sizes_from_cardinality(cardinality_x_expected_dim):
    multiplier = 5.0

    cardinality, expected_dim = cardinality_x_expected_dim
    dim = get_embedding_size_from_cardinality(cardinality, multiplier)
    assert dim == expected_dim


@pytest.mark.parametrize(
    "cardinality_x_expected_dim",
    [
        (1, 8),
        (5, 8),
        (10, 16),
        (20, 16),
        (50, 16),
        (100, 16),
        (500, 24),
        (1000, 32),
        (5000, 48),
        (10000, 56),
        (50000, 80),
        (100000, 96),
        (500000, 136),
        (1000000, 160),
        (5000000, 240),
        (10000000, 288),
    ],
)
def test_get_embedding_size_from_cardinality_multiple_of_8(cardinality_x_expected_dim):
    multiplier = 5.0

    cardinality, expected_dim = cardinality_x_expected_dim
    dim = get_embedding_size_from_cardinality(cardinality, multiplier, ensure_multiple_of_8=True)
    assert dim == expected_dim
