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

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

from merlin.models.tf.losses.base import LossRegistryMixin


@LossRegistryMixin.registry.register_with_multiple_names("sparse_categorical_crossentropy", "sce")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SparseCategoricalCrossEntropy(SparseCategoricalCrossentropy, LossRegistryMixin):
    """Extends `tf.keras.losses.SparseCategoricalCrossentropy` by making
    `from_logits=True` by default (in this case an optimized `softmax` activation
    is applied within this loss, you should not include `softmax` activation
    manually in the output layer). It also adds support to
    the loss_registry, so that the loss can be defined by the user
    by a string alias name.
    """

    def __init__(self, from_logits=True, **kwargs):
        super().__init__(from_logits=from_logits, **kwargs)


@LossRegistryMixin.registry.register_with_multiple_names("categorical_crossentropy", "ce")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalCrossEntropy(CategoricalCrossentropy, LossRegistryMixin):
    """Extends `tf.keras.losses.SparseCategoricalCrossentropy` by making
    `from_logits=True` by default (in this case an optimized `softmax` activation
    is applied within this loss, you should not include `softmax` activation
    manually in the output layer). It also adds support to
    the loss_registry, so that the loss can be defined by the user
    by a string alias name.
    """

    def __init__(self, from_logits=True, **kwargs):
        super().__init__(from_logits=from_logits, **kwargs)
