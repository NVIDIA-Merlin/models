from typing import Dict, Sequence, Union

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, type_spec
from tensorflow.python.framework.composite_tensor import CompositeTensor
from tensorflow.python.util import nest

from merlin.schema import ColumnSchema, Schema, Tags


class TypeSpec(type_spec.TypeSpec):
    """A generic CompositeTensor TypeSpec, used for constructing tests."""

    def __init__(self, component_specs, metadata=None):
        self.component_specs = component_specs
        self.metadata = metadata

    def _serialize(self):
        return (self.component_specs, self.metadata)

    def _to_components(self, value):
        return value.components

    def _from_components(self, tensor_list):
        return FeatureCollection(tensor_list, self.metadata)

    @property
    def value_type(self):
        return FeatureCollection

    @property
    def _component_specs(self):
        return self.component_specs


class FeatureCollection(CompositeTensor):
    """
    A collection of features containing their schemas and data.
    """

    def __init__(self, schema: Schema, values: Dict[str, tf.Tensor]):
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

    def value(self, name: str) -> tf.Tensor:
        return self.values[name]

    def col(self, name: str) -> ColumnSchema:
        return self.schema.column_schemas[name]

    def select_value_by_tag(self, tag):
        selected = self.select_by_tag(tag)

        if not len(selected.values) == 1:
            raise ValueError("")

        return list(selected.values.values())[0]

    def _type_spec(self):
        component_specs = nest.map_structure(type_spec.type_spec_from_value, self.values)
        return TypeSpec(component_specs, self.schema)

    def _shape_invariant_to_type_spec(self, shape):
        rank = tensor_shape.dimension_value(shape[0])

        return TypeSpec(tensor_shape.unknown_shape(rank))

    @property
    def shape(self) -> Dict[str, tf.TensorShape]:
        return {key: val.shape for key, val in self.values.items()}
