#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from typing import Dict, Optional, Union, Sequence

import tensorflow as tf
from merlin.schema import Schema, ColumnSchema, Tags
from tensorflow.python.data.util import nest
from tensorflow.python.framework import type_spec, tensor_shape
from tensorflow.python.framework.composite_tensor import CompositeTensor

# from merlin.models.config.schema import FeatureCollection


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


class Context(FeatureCollection):
    TARGET_NANE = "__target__"

    def __init__(
            self,
            schema: Schema,
            features: Dict[str, tf.Tensor],
            targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None,
    ):
        values = {**features}
        if targets is not None:
            if isinstance(targets, dict):
                values.update(targets)
                self._target_names = list(targets.keys())
            else:
                values[Context.TARGET_NANE] = targets
                self._target_names = [Context.TARGET_NANE]

        super(Context, self).__init__(schema, values)
        self._feature_names = list(features.keys())

    @property
    def feature_collection(self) -> FeatureCollection:
        return FeatureCollection(
            self.schema.select_by_name(self._feature_names),
            self.features
        )

    @property
    def features(self) -> Dict[str, tf.Tensor]:
        return {name: self.values[name] for name in self._feature_names}

    @property
    def targets(self) -> Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]]:
        if getattr(self, "_target_names", None) is None:
            return None

        return {name: self.values[name] for name in self._target_names}

    @property
    def target_collection(self) -> FeatureCollection:
        return FeatureCollection(
            self.schema.select_by_name(self._target_names),
            self.targets
        )

    def with_targets(self, targets) -> "Context":
        return Context(self.schema, self.features, targets)


class FeatureContext:
    def __init__(self, features: FeatureCollection, mask: tf.Tensor = None):
        self.features = features
        self._mask = mask

    @property
    def mask(self):
        if self._mask is None:
            raise ValueError("The mask is not stored, " "please make sure that a mask was set")
        return self._mask

    @mask.setter
    def mask(self, mask: tf.Tensor):
        self._mask = mask
