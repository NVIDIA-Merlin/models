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

from typing import Union

import tensorflow as tf

from merlin.models.utils.registry import Registry, RegistryMixin

LossType = Union[str, tf.keras.losses.Loss]

loss_registry: Registry = Registry.class_registry("tf.losses")

# Registering keras pointwise losses
loss_registry.register_with_multiple_names("mean_squared_error", "mse")(
    tf.keras.losses.MeanSquaredError
)
loss_registry.register_with_multiple_names("binary_crossentropy", "bce")(
    tf.keras.losses.BinaryCrossentropy
)


class LossRegistryMixin(RegistryMixin["LossRegistryMixin"]):
    registry = loss_registry


# TODO: Override the __init__ of Losses to accept temperature and override
# call() to apply the temperature (logits/temperature) of logits before
# computing the metrics. Rename `PredictionsScaler` to `TensorScaler` for
# potential other usages
