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
from tensorflow.python.feature_column import feature_column_v2 as fc


def _make_categorical_embedding(name, vocab_size, embedding_dim):
    column = tf.feature_column.categorical_column_with_identity(name, vocab_size)
    if embedding_dim is None:
        return tf.feature_column.indicator_column(column)
    else:
        return tf.feature_column.embedding_column(column, embedding_dim)
