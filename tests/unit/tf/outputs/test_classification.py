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
from merlin.models.tf.outputs.classification import CategoricalTarget, EmbeddingTablePrediction
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_binary_output(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.BinaryOutput("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "loss_batch",
        "precision",
        "recall",
        "binary_accuracy",
        "auc",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("use_output_block", [True, False])
def test_binary_output_two_tasks(ecommerce_data: Dataset, run_eagerly, use_output_block):
    if use_output_block:
        output_block = mm.OutputBlock(ecommerce_data.schema)
    else:
        output_block = mm.ParallelBlock(mm.BinaryOutput("click"), mm.BinaryOutput("conversion"))

    model = mm.Model(mm.InputBlock(ecommerce_data.schema), mm.MLPBlock([8]), output_block)

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(ecommerce_data, batch_size=50))

    assert set(metrics.keys()) == {
        "loss",
        "loss_batch",
        "click/binary_output_loss",
        "click/binary_output/precision",
        "click/binary_output/recall",
        "click/binary_output/binary_accuracy",
        "click/binary_output/auc",
        "conversion/binary_output_loss",
        "conversion/binary_output/precision",
        "conversion/binary_output/recall",
        "conversion/binary_output/binary_accuracy",
        "conversion/binary_output/auc",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_output(sequence_testing_data: Dataset, run_eagerly):
    dataloader, schema = testing_utils.loader_for_last_item_prediction(sequence_testing_data)
    model = mm.Model(
        mm.InputBlockV2(schema),
        mm.MLPBlock([8]),
        mm.CategoricalOutput(schema["item_id_seq"]),
    )

    _, history = testing_utils.model_test(model, dataloader, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "loss_batch",
        "ndcg_at_10",
        "precision_at_10",
        "map_at_10",
        "mrr_at_10",
        "recall_at_10",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_last_item_prediction(sequence_testing_data: Dataset, run_eagerly):
    dataloader, schema = testing_utils.loader_for_last_item_prediction(sequence_testing_data)
    embeddings = mm.Embeddings(
        schema,
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
            mm.InputBlockV2(schema, categorical=embeddings),
            mm.MLPBlock([32]),
            mm.CategoricalOutput(target),
        )
        testing_utils.model_test(model, dataloader, run_eagerly=run_eagerly)
