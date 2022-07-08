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
from typing import Dict, Generic, List, NamedTuple, Optional, Sequence, TypeVar, Union

import tensorflow as tf
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.composite_tensor import CompositeTensor

from merlin.models.config.schema import FeatureCollection
from merlin.models.tf.utils.composite_tensor_utils import AutoCompositeTensorTypeSpec
from merlin.models.utils import schema_utils
from merlin.schema import Schema

TensorLike = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, CompositeTensor]
ValueT = TypeVar("ValueT", TensorLike, Dict[str, TensorLike], Sequence[TensorLike])


@type_spec.register("merlin.PredictionContextTypeSpec")
class PredictionContextTypeSpec(AutoCompositeTensorTypeSpec):
    @property
    def value_type(self):
        return PredictionContext

    def _serialize(self):
        output = super()._serialize()
        output[1]["schema"] = schema_utils.schema_to_tensorflow_metadata_json(output[1]["schema"])

        return output

    @classmethod
    def _deserialize(cls, encoded):
        encoded[1]["schema"] = schema_utils.tensorflow_metadata_json_to_schema(encoded[1]["schema"])

        return super()._deserialize(encoded)


@type_spec.register("merlin.PredictionTensorTypeSpec")
class PredictionTensorTypeSpec(AutoCompositeTensorTypeSpec):
    @property
    def value_type(self):
        return PredictionTensor


class PredictionContextShape(NamedTuple):
    features: Dict[str, tf.TensorShape]
    targets: Optional[Union[tf.TensorShape, Dict[str, tf.TensorShape]]] = None


class PredictionContext(CompositeTensor):
    def __init__(
        self,
        schema: Schema,
        features: Dict[str, tf.Tensor],
        targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None,
        mask: tf.Tensor = None,
        training: bool = False,
        testing: bool = False,
    ):
        self.features = features
        self.schema = schema
        self.targets = targets
        self.mask = mask
        self.training = training
        self.testing = testing

    @property
    def feature_collection(self) -> FeatureCollection:
        return FeatureCollection(self.schema.select_by_name(self.feature_names), self.features)

    @property
    def target_collection(self) -> FeatureCollection:
        return FeatureCollection(self.schema.select_by_name(self.targets_names), self.targets)

    def with_targets(self, targets) -> "PredictionContext":
        return PredictionContext(
            self.schema, self.features, targets, self.mask, self.training, self.testing
        )

    def with_mask(self, mask) -> "PredictionContext":
        return PredictionContext(
            self.schema, self.features, self.targets, mask, self.training, self.testing
        )

    @property
    def feature_names(self) -> List[str]:
        return list(self.features.keys())

    @property
    def targets_names(self) -> List[str]:
        return list(self.targets.keys()) if self.targets else []

    @property
    def _type_spec(self):
        return PredictionContextTypeSpec.from_instance(self)

    @property
    def shape(self) -> PredictionContextShape:
        targets = getattr(self.targets, "shape", None)
        if isinstance(self.targets, dict):
            targets = {key: val.shape for key, val in self.targets.items()}

        return PredictionContextShape(
            features={key: val.shape for key, val in self.features.items()}, targets=targets
        )

    def __eq__(self, other):
        return self._type_spec == other._type_spec

    def __repr__(self):
        return f"Context({self.features})"

    def to_call_dict(self):
        output = {
            "features": self.features,
            "training": self.training,
            "testing": self.testing,
        }

        if self.training or self.testing:
            output["mask"] = self.mask
            output["targets"] = self.targets

        # if self.targets is not None:
        #     output["targets"] = self.targets

        return output


type_spec.register_type_spec_from_value_converter(
    PredictionContext, PredictionContextTypeSpec.from_instance
)


class PredictionTensorShape(NamedTuple):
    value: tf.TensorShape
    context: PredictionContextShape


class PredictionTensor(CompositeTensor, Generic[ValueT]):
    def __init__(self, value: ValueT, context: PredictionContext):
        self.value = value
        self.context = context

    @property
    def _type_spec(self):
        return PredictionTensorTypeSpec.from_instance(self)

    def __eq__(self, other):
        return self._type_spec == other._type_spec

    def __repr__(self):
        return f"ContextTensor({self.value})"

    @property
    def shape(self) -> PredictionTensorShape:
        value = getattr(self.value, "shape", None)
        if isinstance(self.value, dict):
            value = {key: val.shape for key, val in self.value.items()}

        return PredictionTensorShape(value=value, context=self.context.shape)


type_spec.register_type_spec_from_value_converter(
    PredictionTensor, PredictionTensorTypeSpec.from_instance
)

# class FeatureContext:
#     def __init__(self, features: FeatureCollection, mask: tf.Tensor = None):
#         self.features = features
#         self._mask = mask
#
#     @property
#     def mask(self):
#         if self._mask is None:
#             raise ValueError("The mask is not stored, " "please make sure that a mask was set")
#         return self._mask
#
#     @mask.setter
#     def mask(self, mask: tf.Tensor):
#         self._mask = mask
