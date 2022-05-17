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
import tensorflow as tf

import merlin.models.tf as ml


@pytest.mark.parametrize(
    "loss",
    [
        # Pairwise losses
        "bpr",
        ml.losses.BPRLoss(),
        "bpr-max",
        ml.losses.BPRmaxLoss(reg_lambda=1.0),
        "top1",
        ml.losses.TOP1Loss(),
        "top1_v2",
        ml.losses.TOP1v2Loss(),
        "top1-max",
        ml.losses.TOP1maxLoss(),
        "logistic",
        ml.losses.LogisticLoss(),
        "hinge",
        ml.losses.HingeLoss(),
        # Listwise losses
        "sparse_categorical_crossentropy",
        "categorical_crossentropy",
        # Pointwise losses
        "mse",
        "binary_crossentropy",
    ],
)
def test_losses(loss):
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    if loss == "sparse_categorical_crossentropy":
        targets = tf.argmax(targets, axis=1)

    loss = ml.losses.loss_registry.parse(loss)
    loss_output = loss(targets, predictions)
    assert len(tf.shape(loss_output)) == 0
    assert loss_output > 0


def test_pairwise_loss_no_reduction():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr = ml.losses.BPRLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr(targets, predictions)
    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    assert tf.reduce_mean(loss) > 0


def test_pairwise_loss_with_sample_weights():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)
    sample_weights = tf.range(1, 101, dtype=tf.float32)

    bpr_max = ml.losses.BPRLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr_max(targets, predictions)
    loss_with_sampled_weights = bpr_max(targets, predictions, sample_weights)

    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    tf.assert_equal(tf.shape(loss_with_sampled_weights), (batch_size, num_samples - 1))

    tf.assert_equal(loss * tf.expand_dims(sample_weights, -1), loss_with_sampled_weights)

    assert (tf.reduce_mean(loss) * tf.reduce_mean(sample_weights)).numpy() == pytest.approx(
        tf.reduce_mean(loss_with_sampled_weights).numpy(), 0.05
    )


def test_pairwise_loss_multiple_positive():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 2), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 2), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr = ml.losses.BPRLoss()

    with pytest.raises(Exception) as excinfo:
        _ = bpr(targets, predictions)
    assert "The batch contains more examples with more than one positive item" in str(excinfo.value)


def test_pairwise_loss_some_rows_with_no_positive():
    batch_size = 30
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)

    n_rows_without_positive = batch_size - num_samples
    targets = tf.eye(num_rows=batch_size, num_columns=num_samples)

    bpr = ml.losses.BPRLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr(targets, predictions)

    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    tf.assert_greater(tf.reduce_min(loss[:-n_rows_without_positive]), 0.0)
    tf.assert_equal(tf.reduce_sum(loss[-n_rows_without_positive:]), 0.0)


def test_pairwise_loss_with_false_negatives():
    batch_size = 30
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)
    valid_negatives_mask = tf.cast(
        1.0 - tf.eye(num_rows=batch_size, num_columns=num_samples - 1), tf.bool
    )

    bpr = ml.losses.BPRLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr(targets, predictions, valid_negatives_mask=valid_negatives_mask)

    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    # Checks if the loss of false negatives are zero
    tf.assert_equal(tf.reduce_sum(tf.boolean_mask(loss, tf.logical_not(valid_negatives_mask))), 0.0)
    # Checks if the loss of true negatives are non-zero
    tf.assert_greater(tf.reduce_min(tf.boolean_mask(loss, valid_negatives_mask)), 0.0)
