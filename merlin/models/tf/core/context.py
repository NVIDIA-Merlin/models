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

import tensorflow as tf

from merlin.models.config.schema import FeatureCollection


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
