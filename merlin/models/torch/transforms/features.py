from merlin.models.torch.core.base import TabularBlock
from merlin.models.torch.typing import TabularData
from merlin.schema import Schema


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
