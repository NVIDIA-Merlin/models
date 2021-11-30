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

import pytest

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")
test_utils = pytest.importorskip("merlin_models.tf.utils.testing_utils")


def test_binary_classification_head(tf_tabular_features, tf_tabular_data):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = tf_tabular_features.connect(ml.MLPBlock([64]))
    body_2  = body.connect(ml.MLPBlock([64]))

    # simple case
    task = ml.BinaryClassificationTask("target")
    model = body.connect(task)  # model

    # head case
    task_2 = ml.BinaryClassificationTask("target_2", task_block=None)
    model = body.connect_branch(task, task_2)   # model

    tasks = ml.ParallelPredictionBlock.from_schema()
    model = body.connect_branch(*tasks)   # model
    model_2 = body_2.connect_branch(*tasks)   # model

    advanced_model = ml.ParallelPredictionBlock(model, model_2)


    # MMOE
    mmoe = ml.MMOEBlock.from_schema(schema, expert_block=ml.MLPBlock([64]))
    model = body.connect(mmoe)


    head = ml.BinaryClassificationTask("target").to_head(body=tf_tabular_features.connect(ml.MLPBlock([64])))

    test_utils.assert_loss_and_metrics_are_valid(head, tf_tabular_data, targets)


def test_serialization_binary_classification_head(tf_tabular_features, tf_tabular_data):
    targets = {"target": tf.cast(tf.random.uniform((100,), maxval=2, dtype=tf.int32), tf.float32)}

    body = ml.SequentialBlock([tf_tabular_features, ml.MLPBlock([64])])
    task = ml.BinaryClassificationTask("target")
    head = task.to_head(body, tf_tabular_features)

    copy_head = test_utils.assert_serialization(head)
    test_utils.assert_loss_and_metrics_are_valid(copy_head, tf_tabular_data, targets)
