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
from merlin_models.tensorflow.layers import DenseFeatures, LinearFeatures


def channels(numeric_columns, categorical_columns, **kwargs):
    """
  Going to just throw everything in both wide and deep channels
  for now. This isn't, in general, how you would want to do this,
  and you may want to do some more complicated logic on the
  ProblemSchema in order to decide how to funnel stuff.
  """
    deep_embedding_dims = kwargs.get("embedding_dims")
    if deep_embedding_dims is None:
        deep_embedding_dims = {
            col.name: kwargs["embedding_dim"] for col in categorical_columns
        }
    deep_embedding_columns = [
        tf.feature_column.embedding_column(col, deep_embedding_dims[col.name])
        for col in categorical_columns
    ]

    return {
        "wide": categorical_columns + numeric_columns,
        "deep": deep_embedding_columns + numeric_columns,
    }


def architecture(channels, inputs, **kwargs):
    """
  https://arxiv.org/pdf/1606.07792.pdf
  """
    deep_x = DenseFeatures(channels["deep"])(inputs["deep"])
    for dim in kwargs["deep_hidden_dims"]:
        deep_x = tf.keras.layers.Dense(dim, activation="relu")(deep_x)
        # + batchnorm, dropout, whatever...
    deep_x = tf.keras.layers.Dense(1, activation="linear")(deep_x)

    wide_x = LinearFeatures(channels["wide"])(inputs["wide"])
    x = tf.keras.layers.Add()([deep_x, wide_x])
    x = tf.keras.layers.Activation("sigmoid")(x)
    return x
