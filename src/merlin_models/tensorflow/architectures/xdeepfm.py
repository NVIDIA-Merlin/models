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
from merlin_models.tensorflow.layers import DenseFeatures, LinearFeatures, XDeepFmOuterProduct

from . import arch_utils


def channels(numeric_columns, categorical_columns, **kwargs):
    embedding_dim = arch_utils.get_embedding_dim(kwargs)
    embedding_columns = arch_utils.get_embedding_columns(
        categorical_columns, embedding_dim
    )

    # not really clear to me how to use numeric columns in CIN so will
    # only feed them to deep channel
    channels = {"CIN": embedding_columns, "deep": numeric_columns}

    if kwargs["use_wide"]:
        channels["wide"] = numeric_columns + categorical_columns
    return channels


def architecture(channels, inputs, **kwargs):
    """
  https://arxiv.org/pdf/1803.05170.pdf
  """
    embedding_dim = arch_utils.get_embedding_dim(kwargs)

    categorical_embeddings = DenseFeatures(channels["CIN"])(inputs["CIN"])
    continuous_embeddings = DenseFeatures(channels["deep"])(inputs["deep"])

    deep_x = tf.keras.layers.Concatenate(axis=1)([categorical_embeddings, continuous_embeddings])
    for dim in kwargs["deep_hidden_dims"]:
        deep_x = tf.keras.layers.Dense(dim, activation="relu")(deep_x)
    deep_x = tf.keras.layers.Dense(1, activation="linear")(deep_x)

    cin_x0 = tf.keras.layers.Reshape((len(inputs["CIN"]), embedding_dim))(categorical_embeddings)
    cin_x = cin_x0

    sum_pool_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))
    pooled_outputs = []
    for dim in kwargs["cin_hidden_dims"]:
        cin_x = XDeepFmOuterProduct(dim)([cin_x, cin_x0])
        pooled_outputs.append(sum_pool_layer(cin_x))
    cin_x = tf.keras.layers.Concatenate(axis=1)(pooled_outputs)

    activation_inputs = [cin_x, deep_x]
    if "wide" in channels:
        wide_x = LinearFeatures(channels["wide"])(inputs["wide"])
        activation_inputs.append(wide_x)

    x = tf.keras.layers.Concatenate(axis=1)(activation_inputs)
    x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    return x