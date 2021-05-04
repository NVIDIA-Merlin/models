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
from merlin_models.tensorflow.layers import DenseFeatures, DotProductInteraction

from . import arch_utils


def channels(numeric_columns, categorical_columns, **kwargs):
    embedding_dim = arch_utils.get_embedding_dim(kwargs)
    embedding_columns = arch_utils.get_embedding_columns(
        categorical_columns, embedding_dim
    )
    return {"dense": numeric_columns, "fm": embedding_columns}


def architecture(channels, inputs, **kwargs):
    """
  https://arxiv.org/pdf/1906.00091.pdf
  See model description at the bottom of page 3
  """
    # check embedding dim up front
    embedding_dim = arch_utils.get_embedding_dim(kwargs)

    # build inputs and map to dense representations
    fm = DenseFeatures(channels["fm"], aggregation="stack")(inputs["fm"])
    dense = DenseFeatures(channels["dense"], aggregation="concat")(inputs["dense"])

    # 'bottom' or 'dense' MLP
    for dim in kwargs["bottom_mlp_hidden_dims"]:
        dense = tf.keras.layers.Dense(dim, activation="relu")(dense)
        # + dropout, batchnorm, whatever...
    dense = tf.keras.layers.Dense(embedding_dim, activation="relu")(dense)

    dense_fm = tf.keras.layers.Reshape((1, embedding_dim))(dense)
    fm = tf.keras.layers.Concatenate(axis=1)([fm, dense_fm])
    fm = DotProductInteraction()(fm)

    # 'top' or 'output' MLP
    x = tf.keras.layers.Concatenate(axis=-1)([fm, dense])
    for dim in kwargs["top_mlp_hidden_dims"]:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return x


def add_parser_args(parser):
    parser.add_argument(
        "--bottom_mlp_hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help=(
            "Space separated sizes of hidden layers used in bottom MLP, "
            "which transforms the continuous features *before* "
            "matrix factorization"
        ),
    )
    parser.add_argument(
        "--top_mlp_hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help=(
            "Space separated sizes of hidden layers in top MLP, "
            "which transforms the matrix factorized features into "
            "a probability of interaction"
        ),
    )
    return parser

