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

import merlin.models.tf as ml


def test_fm_pairwise_interaction():
    NUM_ROWS = 100
    NUM_FEATS = 10
    EMBED_DIM = 64

    inputs = tf.random.uniform((NUM_ROWS, NUM_FEATS, EMBED_DIM))

    pairwise_interaction = ml.FMPairwiseInteraction()
    outputs = pairwise_interaction(inputs)

    assert list(outputs.shape) == [NUM_ROWS, EMBED_DIM]
