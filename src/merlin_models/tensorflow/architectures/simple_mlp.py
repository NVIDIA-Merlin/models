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
from merlin_models.tensorflow.layers import DenseFeatures


def channels(numeric_columns, categorical_columns, **kwargs):
    embedding_dims = kwargs.get("embedding_dims")
    if embedding_dims is None:
        embedding_dims = {
            col.name: kwargs["embedding_dim"] for col in categorical_columns
        }

    embedding_columns = []
    for column in categorical_columns:
        dim = embedding_dims[column.name]
        embedding_columns.append(tf.feature_column.embedding_column(column, dim))
    return {"mlp": numeric_columns + embedding_columns}


def architecture(channels, inputs, **kwargs):
    x = DenseFeatures(channels["mlp"])(inputs["mlp"])
    for dim in kwargs["hidden_dims"]:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        # + batchnorm, dropout, whatever...
    return tf.keras.layers.Dense(1, activation="sigmoid")(x)
