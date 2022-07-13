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

from merlin.models.tf.data_augmentation.misc import ContinuousPowers


def test_continuous_powers():
    NUM_ROWS = 100

    inputs = {
        "cont_feat_1": tf.random.uniform((NUM_ROWS,)),
        "cont_feat_2": tf.random.uniform((NUM_ROWS,)),
    }

    powers = ContinuousPowers()

    outputs = powers(inputs)

    assert len(outputs) == len(inputs) * 3
    for key in inputs:
        assert key in outputs
        assert key + "_sqrt" in outputs
        assert key + "_pow" in outputs

        tf.assert_equal(tf.sqrt(inputs[key]), outputs[key + "_sqrt"])
        tf.assert_equal(tf.pow(inputs[key], 2), outputs[key + "_pow"])
