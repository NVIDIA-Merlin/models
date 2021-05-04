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
from merlin_models.tensorflow.layers import DenseFeatures, LinearFeatures, DotProductInteraction

from . import arch_utils


def channels(numeric_columns, categorical_columns, **kwargs):
    embedding_dim = arch_utils.get_embedding_dim(kwargs)
    embedding_columns = arch_utils.get_embedding_columns(
        categorical_columns, embedding_dim
    )

    channels = {"fm": embedding_columns, "deep": numeric_columns}

    if kwargs["use_wide"]:
        channels["wide"] = categorical_columns + numeric_columns
    return channels


def architecture(channels, inputs, **kwargs):
    """
  https://arxiv.org/pdf/1703.04247.pdf
  Note that this paper doesn't specify how continuous features are
  handled during the FM portion, so I'm just going to exclude them
  and only use them for the deep portion. There's probably better
  ways to handle this, but since this is meant to be a verbatim
  implementation of the paper I'm going to take the simplest
  route possible
  """
    embedding_dim = arch_utils.get_embedding_dim(kwargs)

    fm_embeddings = DenseFeatures(channels["fm"])(inputs["fm"])
    deep_embeddings = DenseFeatures(channels["deep"])(inputs["deep"])

    # Deep portion
    deep_x = tf.keras.layers.Concatenate(axis=1)([fm_embeddings, deep_embeddings])
    for dim in kwargs["hidden_dims"]:
        deep_x = tf.keras.layers.Dense(dim, activation="relu")(deep_x)
        # + batchnorm, dropout, whatever...
    deep_x = tf.keras.layers.Dense(1, activation="linear")(deep_x)

    # FM portion
    fm_shape = (len(inputs["fm"], embedding_dim))
    fm_embeddings = tf.keras.layers.Reshape(fm_shape)(fm_embeddings)
    fm_x = DotProductInteraction()(fm_embeddings)

    # maybe do wide
    activation_inputs = [fm_x, deep_x]
    if "wide" in channels:
        wide_x = LinearFeatures(channels["wide"])(inputs["wide"])
        activation_inputs.append(wide_x)

    # Combine, Sum, Activate
    # note that we can't just add since `fm_x` has dim
    # k(k-1)/2, where k is the number of categorical features
    x = tf.keras.layers.Concatenate(axis=1)(activation_inputs)
    x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    return x