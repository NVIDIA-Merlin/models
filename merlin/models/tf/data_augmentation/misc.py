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

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.tabular import TabularBlock
from merlin.models.tf.typing import TabularData


@Block.registry.register_with_multiple_names("continuous-powers")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContinuousPowers(TabularBlock):
    """Trick from `Deep Neural Networks for YouTube Recommendations`"""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs: TabularData = {}

        for key, val in inputs.items():
            outputs[key] = val
            if len(val.shape) < 2 or (len(val.shape) == 2 and val.shape[1] == 1):
                val_float = tf.cast(val, tf.float32)
                outputs[f"{key}_sqrt"] = tf.sqrt(val_float)
                outputs[f"{key}_pow"] = tf.pow(val_float, 2)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = {}

        for key, val in input_shape.items():
            output_shape[key] = val
            if len(val) < 2 or (len(val) == 2 and val[1] == 1):
                output_shape[f"{key}_sqrt"] = val
                output_shape[f"{key}_squared"] = val

        return output_shape
