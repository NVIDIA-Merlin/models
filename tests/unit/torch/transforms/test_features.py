import pytest

from merlin.models.torch.transforms.features import Filter
from merlin.schema import Schema, Tags


class TestFilter:
    def test_init(self):
        filter_obj = Filter(Schema(["a", "b", "c"]))

        assert isinstance(filter_obj, Filter)
        assert filter_obj._feature_names == {"a", "b", "c"}
        assert not filter_obj.exclude
        assert not filter_obj.pop

    def test_schema_setter(self):
        schema = Schema(["col1", "col2", "col3"])
        f = Filter(schema)

        with pytest.raises(ValueError, match="Expected a Schema object, got <class 'str'>"):
            f.schema = "not a schema"

        new_schema = Schema(["col4", "col5", "col6"])
        f.schema = new_schema
        assert f.schema == new_schema

    def test_select_by_name(self):
        schema = Schema(["col1", "col2", "col3"])
        f = Filter(schema)

        assert f.select_by_name("col1")._feature_names == {"col1"}

    def test_select_by_tag(self, user_id_col_schema, item_id_col_schema):
        schema = Schema([user_id_col_schema, item_id_col_schema])
        f = Filter(schema)

        assert f.select_by_tag(Tags.USER_ID)._feature_names == {"user_id"}

    def test_forward(self):
        schema = Schema(["col1", "col2"])
        f = Filter(schema)

        input_data = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}

        expected_output = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

        filtered_data = f(input_data)
        assert filtered_data == expected_output

        f.exclude = True
        expected_output = {"col3": [7, 8, 9]}
        filtered_data = f(input_data)
        assert filtered_data == expected_output

    def test_check_feature(self):
        schema = Schema(["col1", "col2", "col3"])
        f = Filter(schema)

        assert f.check_feature("col1")
        assert f.check_feature("col2")
        assert f.check_feature("col3")
        assert not f.check_feature("col4")

        f.exclude = True
        assert not f.check_feature("col1")
        assert not f.check_feature("col2")
        assert not f.check_feature("col3")
        assert f.check_feature("col4")
