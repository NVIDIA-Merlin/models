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


def test_brute_force_layer():
    from merlin.models.tf.core.prediction import TopKPrediction
    from merlin.models.tf.outputs.topk import BruteForce

    num_candidates = 1000
    num_queries = 100
    top_k = 5

    candidates = tf.random.uniform(shape=(num_candidates, 4), dtype=tf.float32)
    query = tf.random.uniform(shape=(num_queries, 4), dtype=tf.float32)

    wrong_candidates_rank = tf.random.uniform(shape=(num_candidates,), dtype=tf.float32)
    wrong_query_dim = tf.random.uniform(shape=(num_queries, 8), dtype=tf.float32)
    wrong_identifiers_shape = tf.range(num_candidates + 1, dtype=tf.int32)

    brute_force = BruteForce(k=top_k)

    with pytest.raises(Exception) as excinfo:
        brute_force.index(candidates=candidates, identifiers=wrong_identifiers_shape)
    assert "The candidates and identifiers tensors must have the same number of rows " in str(
        excinfo.value
    )

    with pytest.raises(Exception) as excinfo:
        brute_force.index(wrong_candidates_rank)
    assert "candidates must be 2-D tensor (got (1000,))" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        brute_force(query)
    assert "You should call the `index` method first to set the _candidates index." in str(
        excinfo.value
    )

    brute_force.index(candidates=candidates)

    with pytest.raises(Exception) as excinfo:
        brute_force(wrong_query_dim)
    assert "Query and candidates vectors must have the same embedding size" in str(excinfo.value)

    topk_output = brute_force(query)
    assert list(topk_output.scores.shape) == [num_queries, top_k]
    assert list(topk_output.identifiers.shape) == [num_queries, top_k]
    assert isinstance(topk_output, TopKPrediction)

    with pytest.raises(Exception) as excinfo:
        brute_force(query, targets=None, testing=True)
    assert "Targets should be provided during the evaluation mode" in str(excinfo.value)

    new_candidates = tf.random.uniform(shape=(num_candidates, 4), dtype=tf.float32)
    brute_force.index(candidates=new_candidates)
    tf.debugging.assert_none_equal(brute_force._candidates, candidates)
