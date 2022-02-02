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
from tensorflow.keras.losses import Loss


class BPR(Loss):
    """BPR loss. Accepts a single positive items per example"""

    def _check_only_one_positive_label_per_example(self, y_true):
        tf.assert_equal(
            tf.reduce_sum(y_true, axis=1),
            tf.ones_like(y_true[:, 0]),
            message="Only one positive label is allowed per example in this BPR implementation",
        )

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        self._check_only_one_positive_label_per_example(y_true)

        positives_mask = tf.cast(y_true, tf.bool)
        positives_scores = tf.boolean_mask(y_pred, positives_mask)
        sub = tf.expand_dims(positives_scores, -1) - y_pred
        loss = -tf.math.log(tf.nn.sigmoid(sub))
        return loss


class BPR_v2(Loss):
    """BPR loss. Accepts multiple positive items per example"""

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        positives_mask = tf.cast(y_true, tf.bool)

        sub_expanded = tf.expand_dims(y_pred, -1) - tf.tile(
            tf.expand_dims(y_pred, 1), [1, tf.shape(y_pred)[1], 1]
        )
        only_positives_subs = tf.boolean_mask(sub_expanded, positives_mask)
        only_positives_negs_subs = tf.boolean_mask(
            only_positives_subs, tf.logical_not(positives_mask)
        )

        loss = -tf.math.log(tf.nn.sigmoid(only_positives_negs_subs))
        return loss
