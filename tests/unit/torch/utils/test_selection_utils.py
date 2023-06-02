import pytest

from merlin.models.torch.utils.selection_utils import Selectable, select_schema, selection_name
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
        assert output == output_2 == Schema([column])

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

    def test_exception(self):
        with pytest.raises(ValueError, match="is not valid"):
            selection_name(1)


class TestSelectable:
    def test_exception(self):
        selectable = Selectable()

        assert hasattr(selectable, "setup_schema")
        with pytest.raises(NotImplementedError):
            selectable.select(1)
