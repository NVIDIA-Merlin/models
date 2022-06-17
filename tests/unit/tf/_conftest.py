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

import pathlib

import numpy as np
import pytest
import tensorflow as tf

NUM_EXAMPLES = 1000
MAX_CARDINALITY = 100
ASSETS_DIR = pathlib.Path(__file__).parent.parent / "assets"


@pytest.fixture(autouse=True)
def tf_clear_session():
    tf.keras.backend.clear_session()


@pytest.fixture
def tf_con_features():
    features = {}
    keys = "abcdef"

    for key in keys:
        features[key] = tf.random.uniform((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def tf_cat_features():
    features = {}
    keys = [f"cat_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = tf.random.uniform((NUM_EXAMPLES, 1), maxval=MAX_CARDINALITY, dtype=tf.int32)

    return features


@pytest.fixture
def tf_masking_inputs():
    # fixed parameters for tests
    NUM_EXAMPLES = 20
    MAX_LEN = 10
    PAD_TOKEN = 0
    hidden_dim = 16
    features = {}
    # generate random tensors for test
    features["input_tensor"] = tf.convert_to_tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = np.random.randint(1, MAX_CARDINALITY, (NUM_EXAMPLES, MAX_LEN))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    labels = tf.convert_to_tensor(labels)
    features["labels"] = labels
    features["padding_idx"] = PAD_TOKEN
    features["vocab_size"] = MAX_CARDINALITY

    return features


@pytest.fixture
def tf_ranking_metrics_inputs():
    POS_EXAMPLE = 30
    VOCAB_SIZE = 40
    features = {}
    features["scores"] = tf.convert_to_tensor(np.random.uniform(0, 1, (POS_EXAMPLE, VOCAB_SIZE)))
    features["ks"] = tf.convert_to_tensor([1, 2, 3, 5, 10, 20])
    features["labels_one_hot"] = tf.convert_to_tensor(
        np.random.choice(a=[0, 1], size=(POS_EXAMPLE, VOCAB_SIZE))
    )

    features["labels"] = tf.convert_to_tensor(np.random.randint(1, VOCAB_SIZE, (POS_EXAMPLE,)))
    return features
