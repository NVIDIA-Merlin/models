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

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict, Protocol, Union, runtime_checkable

import tensorflow as tf

from merlin.models.tf.typing import TabularData

if TYPE_CHECKING:
    from merlin.models.tf.blocks.corer.base import PredictionOutput


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


class MetricsMixin(abc.ABC):
    """Mixin to use for Keras Layers that can calculate metrics."""

    def calculate_metrics(
        self,
        outputs: PredictionOutput,
        mode: str = "val",
        forward: bool = True,
        training: bool = False,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:

        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `metric_results` method.

        Parameters
        ----------
        outputs: PredictionOutput
            The named tuple containing predictions and targets tensors
        forward: bool, default True

        mode: str, default="val"

        """
        raise NotImplementedError()

    def metric_results(self, mode: str = "val") -> Dict[str, Union[float, tf.Tensor]]:
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
        outputs: PredictionOutput,
        mode: str = "val",
        forward=True,
        training=False,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        ...

    def metric_results(self, mode: str = "val") -> Dict[str, Union[float, tf.Tensor]]:
        ...

    def _set_context(self, context):
        ...
