from typing import Union

import torch

from merlin.models.torch.core.base import TabularBlock
from merlin.models.torch.typing import TabularData
from merlin.models.utils import schema_utils
from merlin.schema import ColumnSchema, Schema, Tags


class Filter(TabularBlock):
    def __init__(
        self, schema: Schema, pre=None, post=None, aggregation=None, exclude=False, pop=False
    ):
        super().__init__(pre=pre, post=post, aggregation=aggregation)
        self.schema = schema
        self.pop = pop
        self.exclude = exclude

    def forward(self, inputs: TabularData) -> TabularData:
        """Filter out features from inputs.

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if self.check_feature(k)}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    @property
    def schema(self) -> Schema:
        return self._schema

    @schema.setter
    def schema(self, value: Schema):
        if not isinstance(value, Schema):
            raise ValueError(f"Expected a Schema object, got {type(value)}")
        self._schema = value
        self._feature_names = set(value.column_names)

    def check_feature(self, feature_name) -> bool:
        if self.exclude:
            return feature_name not in self._feature_names

        return feature_name in self._feature_names

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        del input_schema
        return self.schema


class ToTarget(TabularBlock):
    """Transform columns to targets"""

    def __init__(
        self,
        schema: Schema,
        *target: Union[ColumnSchema, Schema, str, Tags],
        one_hot: bool = False,
        pre=None,
        post=None,
        aggregation=None,
    ):
        super().__init__(pre=pre, post=post, aggregation=aggregation)
        self.schema = schema
        self.target = target
        self.target_columns = schema_utils.select(self.schema, *self.target)
        self.one_hot = one_hot

    def forward(self, inputs: TabularData, targets=None) -> TabularData:
        outputs = {}
        for name in inputs:
            if name not in self.target_columns:
                outputs[name] = inputs[name]
                continue
            if isinstance(targets, dict):
                _target = targets.get(name, inputs[name])
                if self.one_hot:
                    _target = self._to_one_hot(name, _target)
                targets[name] = _target
            else:
                _target = inputs[name]
                if self.one_hot:
                    _target = self._to_one_hot(name, _target)
                targets = _target

        return outputs

    def _to_one_hot(self, name, target):
        num_classes = schema_utils.categorical_cardinalities(self.schema)[name]
        one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        return one_hot.squeeze()

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        output_column_schemas = {}
        for col_name, col_schema in input_schema.column_schemas.items():
            if col_name in self.target_columns:
                output_column_schemas[col_name] = col_schema.with_tags(Tags.TARGET)
            else:
                output_column_schemas[col_name] = col_schema

        return Schema(column_schemas=output_column_schemas)
