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
from merlin.io import Dataset

targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}


def test_binary_classification_head(testing_data: Dataset):
    from merlin.models.tf.utils import testing_utils

    body = ml.InputBlock(testing_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(ml.BinaryClassificationTask("target"))

    batch = (ml.sample_batch(testing_data, batch_size=100, include_targets=False), targets)
    testing_utils.assert_loss_and_metrics_are_valid(model, batch)


def test_serialization_binary_classification_head(testing_data: Dataset):
    from merlin.models.tf.utils import testing_utils

    body = ml.InputBlock(testing_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(ml.BinaryClassificationTask("target"))

    copy_model = testing_utils.assert_serialization(model)
    batch = (ml.sample_batch(testing_data, batch_size=100, include_targets=False), targets)
    testing_utils.assert_loss_and_metrics_are_valid(copy_model, batch)
