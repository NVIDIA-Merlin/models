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
import abc
from typing import Dict, Protocol, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin_standard_lib import Tag

from ..typing import TabularData


class LossMixin(abc.ABC):
    """Mixin to use for Keras Layers that can calculate a loss."""

    def compute_loss(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        compute_metrics=True,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Compute the loss on a batch of data.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        training: bool, default=False
        """
        raise NotImplementedError()


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ModelContext(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shared_variables: Dict[str, tf.Variable] = {}
        self._feature_blocks = []

    def call(self, inputs, training=None, **kwargs):
        for block in self._feature_blocks:
            block.call_features(inputs, training=training)

        return {}

    def register_variable(self, name: str, variable: tf.Variable):
        self._shared_variables[name] = variable

    @property
    def variables(self) -> Dict[str, tf.Variable]:
        return self._shared_variables

    @property
    def feature_shapes(self):
        return self._feature_shapes

    def get_embedding(self, name: Union[str, Tag]) -> tf.Variable:
        return self._shared_variables[f"{name}/embedding"]

    def _merge(self, other: "ModelContext"):
        self._shared_variables.update(other._shared_variables)
        self._feature_blocks = list(set(self._feature_blocks + other._feature_blocks))

    def compute_output_shape(self, input_shape):
        self._feature_shapes = input_shape

        return {}


class ContextMixin:
    @property
    def context(self) -> ModelContext:
        if not hasattr(self, "_context"):
            self._context = ModelContext()

        return self._context

    def _set_context(self, context: ModelContext):
        if hasattr(self, "_context"):
            context._merge(self._context)
        self._context = context


class MetricsMixin(abc.ABC):
    """Mixin to use for Keras Layers that can calculate metrics."""

    def calculate_metrics(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `metric_results` method.

        Parameters
        ----------
        inputs: Union[tf.Tensor, TabularData]
            TODO
        targets: Union[tf.Tensor, TabularData]
            TODO
        forward: bool, default True

        mode: str, default="val"

        """
        raise NotImplementedError()

    def metric_results(self, mode: str = None) -> Dict[str, Union[float, tf.Tensor]]:
        """Returns the current state of each metric.

        The state is typically updated each batch by calling the `calculate_metrics` method.

        Parameters
        ----------
        mode: str, default="val"

        Returns
        -------
        Dict[str, Union[float, tf.Tensor]]
        """
        raise NotImplementedError()

    def reset_metrics(self):
        """Reset all metrics."""
        raise NotImplementedError()


@runtime_checkable
class ModelLikeBlock(Protocol):
    def compute_loss(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        compute_metrics=True,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        ...

    def calculate_metrics(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        mode: str = "val",
        forward=True,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        ...

    def metric_results(self, mode: str = None) -> Dict[str, Union[float, tf.Tensor]]:
        ...

    def __call__(self, inputs, **kwargs):
        ...

    def _set_context(self, context: ModelContext):
        ...
