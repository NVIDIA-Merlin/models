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

import functools

import tensorflow as tf


def get_embedding_dim(kwargs):
    """
  utility function for getting the embedding dimzension of a model
  that requires that all embeddings have a uniform dimension from
  a set of run kwarg. If `embedding_dims` is used instead of
  `embedding_dim`, check to make sure that they all match
  """
    embedding_dim = kwargs.get("embedding_dim")
    if embedding_dim is None:
        embedding_dims = list(kwargs.get("embedding_dims").values())
        embedding_dim = embedding_dims[0]
        assert all([dim == embedding_dim for dim in embedding_dims[1:]])
    return embedding_dim


def get_embedding_columns(categorical_columns, embedding_dim):
    """
  utility function for building a set of embedding_columns of the
  with the same dimension
  """
    make_embedding_column = functools.partial(
        tf.feature_column.embedding_column, dimension=embedding_dim
    )
    return list(map(make_embedding_column, categorical_columns))
