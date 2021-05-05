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

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
models = pytest.importorskip("merlin_models.tensorflow.models")


def test_wide_and_deep_construction():
    scalar_continuous = tf.feature_column.numeric_column("scalar_continuous", (1,))
    vector_continuous = tf.feature_column.numeric_column("vector_continuous", (128,))
    one_hot = tf.feature_column.categorical_column_with_identity("one_hot", 100)
    multi_hot = tf.feature_column.categorical_column_with_identity("multi_hot", 5)

    numeric_columns = [scalar_continuous, vector_continuous]
    categorical_columns = [one_hot, multi_hot]

    embedding_dims = 512
    hidden_dims = [512, 256, 128]

    model = models.WideAndDeep(
        numeric_columns, categorical_columns, embedding_dims, hidden_dims
    )

    model.compile("sgd", "binary_crossentropy")
