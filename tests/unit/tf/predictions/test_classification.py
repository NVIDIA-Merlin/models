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

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.predictions.classification import CategoricalTarget, EmbeddingTablePrediction
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_binary_prediction_block(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.BinaryPrediction("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "click/binary_prediction/precision",
        "click/binary_prediction/recall",
        "click/binary_prediction/binary_accuracy",
        "click/binary_prediction/auc",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_prediction_block(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema
    schema["item_category"] = schema["item_category"].with_tags(
        schema["item_category"].tags + "target"
    )
    ecommerce_data.schema = schema
    model = mm.Model(
        mm.InputBlock(schema),
        mm.MLPBlock([8]),
        mm.CategoricalPrediction(
            target_layer=schema["item_category"], target_name="item_category", max_num_samples=20
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "item_category/categorical_prediction/merlin.models>ndcg_at_10",
        "item_category/categorical_prediction/merlin.models>_precision_at_10",
        "item_category/categorical_prediction/merlin.models>_avg_precision_at_10",
        "item_category/categorical_prediction/merlin.models>mrr_at_10",
        "item_category/categorical_prediction/merlin.models>_recall_at_10",
        "regularization_loss",
    }


# TODO Fix graph model
@pytest.mark.parametrize("run_eagerly", [True])
def test_next_item_prediction(sequence_testing_data: Dataset, run_eagerly):
    def last_interaction_as_target(inputs, targets):
        inputs = mm.AsRaggedFeatures()(inputs)
        items = inputs["item_id_seq"]
        _items = items[:, :-1]
        targets = items[:, -1:].flat_values
        inputs["item_id_seq"] = _items
        return inputs, targets

    schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    sequence_testing_data.schema = schema
    dataloader = BatchedDataset(sequence_testing_data, batch_size=50)
    dataloader = dataloader.map(last_interaction_as_target)

    embeddings = mm.Embeddings(
        sequence_testing_data.schema,
        sequence_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    )

    predictions = [
        schema.select_by_name("item_id_seq"),
        schema["item_id_seq"],
        CategoricalTarget(schema["item_id_seq"]),
        embeddings["item_id_seq"],
        EmbeddingTablePrediction(embeddings["item_id_seq"]),
    ]

    for target in predictions:
        model = mm.Model(
            mm.InputBlockV2(
                sequence_testing_data.schema,
                embeddings=embeddings,
                aggregation=None,
            ),
            mm.MLPBlock([32]),
            mm.CategoricalPrediction(
                target_layer=target, target_name="item_id_seq", max_num_samples=20
            ),
        )
        _, history = testing_utils.model_test(model, dataloader, run_eagerly=run_eagerly)
