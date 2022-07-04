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
from typing import Dict, Generic, NamedTuple, Optional, Sequence, TypeVar, Union

import tensorflow as tf
from tensorflow.python.framework.composite_tensor import CompositeTensor

from merlin.models.tf.utils.composite_tensor_utils import AutoCompositeTensorTypeSpec
from merlin.schema import ColumnSchema, Schema, Tags

# from merlin.models.config.schema import FeatureCollection


# class TypeSpec(type_spec.TypeSpec):
#     """A generic CompositeTensor TypeSpec, used for constructing tests."""
#
#     def __init__(self, component_specs, metadata=None):
#         self.component_specs = component_specs
#         self.metadata = metadata
#
#     def _serialize(self):
#         return (self.component_specs, self.metadata)
#
#     def _to_components(self, value):
#         return value.components
#
#     def _from_components(self, tensor_list):
#         return FeatureCollection(tensor_list, self.metadata)
#
#     @property
#     def value_type(self):
#         return FeatureCollection
#
#     @property
#     def _component_specs(self):
#         return self.component_specs


class FeatureCollectionTypeSpec(AutoCompositeTensorTypeSpec):
    @property
    def value_type(self):
        return FeatureCollection


class ContextTypeSpec(AutoCompositeTensorTypeSpec):
    @property
    def value_type(self):
        return Context

    # def __hash__(self):
    #     return hash("1")


class ContextTensorTypeSpec(AutoCompositeTensorTypeSpec):
    @property
    def value_type(self):
        return ContextTensor


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

    # def _type_spec(self):
    #     component_specs = nest.map_structure(type_spec.type_spec_from_value, self.values)
    #     return TypeSpec(component_specs, self.schema)
    #
    # def _shape_invariant_to_type_spec(self, shape):
    #     rank = tensor_shape.dimension_value(shape[0])
    #
    #     return TypeSpec(tensor_shape.unknown_shape(rank))

    @property
    def _type_spec(self):
        return FeatureCollectionTypeSpec.from_instance(self, ("schema",), ())

    @property
    def shape(self) -> Dict[str, tf.TensorShape]:
        return {key: val.shape for key, val in self.values.items()}


class ContextShape(NamedTuple):
    features: Dict[str, tf.TensorShape]
    targets: Optional[Union[tf.TensorShape, Dict[str, tf.TensorShape]]]


class Context(CompositeTensor):
    def __init__(
        self,
        schema: Schema,
        features: Dict[str, tf.Tensor],
        targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None,
    ):
        self.features = features
        self.schema = schema
        self.targets = targets

    @property
    def feature_collection(self) -> FeatureCollection:
        return FeatureCollection(self.schema.select_by_name(self._feature_names), self.features)

    # @property
    # def features(self) -> Dict[str, tf.Tensor]:
    #     return {**self._features}

    # @property
    # def targets(self) -> Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]]:
    #     if getattr(self, "_target_names", None) is None:
    #         return None
    #
    #     return {name: self.values[name] for name in self._target_names}

    @property
    def target_collection(self) -> FeatureCollection:
        return FeatureCollection(self.schema.select_by_name(self._target_names), self.targets)

    def with_targets(self, targets) -> "Context":
        return Context(self.schema, self.features, targets)

    @property
    def _type_spec(self):
        return ContextTypeSpec.from_instance(self)

    @property
    def shape(self) -> ContextShape:
        targets = getattr(self.targets, "shape", None)
        if isinstance(self.targets, dict):
            targets = {key: val.shape for key, val in self.targets.items()}

        return ContextShape(
            features={key: val.shape for key, val in self.features.items()}, targets=targets
        )

    def __eq__(self, other):
        return self._type_spec == other._type_spec

    def __repr__(self):
        return f"Context({self.features})"


class ContextTensorShape(NamedTuple):
    value: tf.TensorShape
    context: ContextShape


ValueT = TypeVar(
    "ValueT", tf.Tensor, tf.SparseTensor, tf.RaggedTensor, Dict[str, tf.Tensor], Sequence[tf.Tensor]
)


class ContextTensor(CompositeTensor, Generic[ValueT]):
    def __init__(self, value: ValueT, context: Context):
        self.value = value
        self.context = context

    @property
    def _type_spec(self):
        return ContextTensorTypeSpec.from_instance(self)

    def __eq__(self, other):
        return self._type_spec == other._type_spec

    def __repr__(self):
        return f"ContextTensor({self.value})"

    @property
    def shape(self) -> ContextTensorShape:
        value = getattr(self.value, "shape", None)
        if isinstance(self.value, dict):
            value = {key: val.shape for key, val in self.value.items()}

        return ContextTensorShape(value=value, context=self.context.shape)


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
