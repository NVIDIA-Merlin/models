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

import merlin_models.tf as ml


def test_bpr():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    targets = tf.concat(
        [
            tf.ones(shape=(batch_size, 1), dtype=tf.float32),
            tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32),
        ],
        axis=1,
    )

    bpr = ml.BPR()  # (reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr(targets, predictions)

    bpr_v2 = ml.BPR_v2()  # (reduction=tf.keras.losses.Reduction.NONE)
    loss_v2 = bpr_v2(targets, predictions)
    assert loss == loss_v2
