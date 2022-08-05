#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Union

from merlin.schema import ColumnSchema, Schema, Tags


class SchemaMixin:
    REQUIRES_SCHEMA = False

    def set_schema(self, schema=None):
        self.check_schema(schema=schema)

        if schema and not getattr(self, "_schema", None):
            self._schema = schema

        return self

    @property
    def schema(self) -> Schema:
        if not self.has_schema:
            raise ValueError(f"{self.__class__.__name__} does not have a schema.")

        return self._schema

    @property
    def has_schema(self):
        return getattr(self, "_schema", None) is not None

    @schema.setter  # type: ignore
    def schema(self, value):
        if value:
            self.set_schema(value)
        else:
            self._schema = value

    def check_schema(self, schema=None):
        if self.REQUIRES_SCHEMA and not getattr(self, "_schema", None) and not schema:
            raise ValueError(f"{self.__class__.__name__} requires a schema.")

    def __call__(self, *args, **kwargs):
        self.check_schema()

        return super().__call__(*args, **kwargs)

    def _maybe_set_schema(self, input, schema):
        if input and getattr(input, "set_schema", None):
            input.set_schema(schema)

    def get_item_ids_from_inputs(self, inputs):
        return inputs[self.schema.select_by_tag(Tags.ITEM_ID).first.name]

    def get_padding_mask_from_item_id(self, inputs, pad_token=0):
        item_id_inputs = self.get_item_ids_from_inputs(inputs)
        if len(item_id_inputs.shape) != 2:
            raise ValueError(
                "To extract the padding mask from item id tensor "
                "it is expected to have 2 dims, but it has {} dims.".format(item_id_inputs.shape)
            )
        return self.get_item_ids_from_inputs(inputs) != pad_token


def requires_schema(module):
    module.REQUIRES_SCHEMA = True

    return module


@dataclass(frozen=True)
class Feature:
    """
    A feature containing its schema and data.
    """

    schema: ColumnSchema
    value: Any


class FeatureCollection:
    """
    A collection of features containing their schemas and data.
    """

    def __init__(self, schema: Schema, values: Dict[str, Any]):
        self.values = values
        self.schema = schema

    def with_schema(self, schema: Schema) -> "FeatureCollection":
        """
        Create a new FeatureCollection with the same data and an updated Schema.

        Parameters
        ----------
        schema : Schema
            Schema to be applied to FeatureCollection

        Returns
        -------
        FeatureCollection
            New collection of features with updated Schema
        """
        return FeatureCollection(schema, self.values)

    def select_by_name(self, names: Union[str, Sequence[str]]) -> "FeatureCollection":
        """
        Create a new FeatureCollection with only the features that match the provided names.

        Parameters
        ----------
        names : string, [string]
            Names of the features to select.

        Returns
        -------
        FeatureCollection
            A collection of the features that match the provided names
        """
        sub_schema = self.schema.select_by_name(names)
        sub_values = {name: self.values[name] for name in sub_schema.column_names}

        return FeatureCollection(sub_schema, sub_values)

    def select_by_tag(
        self, tags: Union[str, Tags, Sequence[str], Sequence[Tags]]
    ) -> "FeatureCollection":
        """
        Create a new FeatureCollection with only the features that match the provided tags.

        Parameters
        ----------
        tags: Union[str, Tags, Sequence[str], Sequence[Tags]]
            Tags or tag strings of the features to select
        Returns
        -------
        FeatureCollection
            A collection of the features that match the provided tags
        """
        sub_schema = self.schema.select_by_tag(tags)
        sub_values = {name: self.values[name] for name in sub_schema.column_names}

        return FeatureCollection(sub_schema, sub_values)

    def __getitem__(self, feature_name: str) -> Feature:
        return Feature(self.schema.column_schemas[feature_name], self.values[feature_name])
