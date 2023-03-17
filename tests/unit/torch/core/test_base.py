import pytest
import torch
import torch.nn as nn

from merlin.models.torch.core.base import Block, Filter, TabularBlock
from merlin.schema import Schema


class TestBlock:
    def test_no_pre_post(self):
        block = Block()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)

        assert torch.equal(inputs, outputs)

    def test_pre(self):
        pre = nn.Linear(2, 3)
        block = Block(pre=pre)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = pre(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_post(self):
        post = nn.Linear(2, 3)
        block = Block(post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = post(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_pre_post(self):
        pre = nn.Linear(2, 3)
        post = nn.Linear(3, 4)
        block = Block(pre=pre, post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = pre(inputs)
        expected_outputs = post(expected_outputs)

        assert torch.equal(outputs, expected_outputs)


class TestTabularBlock:
    def test_no_pre_post_aggregation(self):
        block = TabularBlock()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)

        assert torch.equal(inputs, outputs)

    def test_aggregation(self):
        aggregation = nn.Linear(2, 3)
        block = TabularBlock(aggregation=aggregation)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = aggregation(inputs)

        assert torch.equal(outputs, expected_outputs)


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
