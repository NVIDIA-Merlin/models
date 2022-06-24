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

import merlin.models.tf as ml
from merlin.datasets.synthetic import generate_data
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(ecommerce_data: Dataset, run_eagerly):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_from_block(ecommerce_data: Dataset, run_eagerly):
    embedding_options = ml.EmbeddingOptions(embedding_dim_default=32)
    model = ml.Model.from_block(
        ml.MLPBlock([64]),
        ecommerce_data.schema,
        prediction_tasks=ml.BinaryClassificationTask("click"),
        embedding_options=embedding_options,
    )

    assert all(
        [f.table.dim == 32 for f in list(model.block.inputs["categorical"].feature_config.values())]
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


def test_block_from_model_with_input(ecommerce_data: Dataset):
    inputs = ml.InputBlock(ecommerce_data.schema)
    block = inputs.connect(ml.MLPBlock([64]))

    with pytest.raises(ValueError) as excinfo:
        ml.Model.from_block(
            block,
            ecommerce_data.schema,
            input_block=inputs,
        )
    assert "The block already includes an InputBlock" in str(excinfo.value)


class MetricsLogger(tf.keras.callbacks.Callback):
    """Callback to keep track of the metrics returned on each step in every epoch.

    The epoch_logs attribute is a dictionary containing a list of all metrics after each batch.

    For example, after training for 1 epoch with 3 batches (steps).
    epoch_logs might look something like the following:
    {
        0: [
            {"auc": 0.599},
            {"auc": 0.515},
            {"auc": 0.450},
        ]
    }
    """

    def __init__(self):
        super().__init__()
        self.epoch_logs = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_logs[epoch] = []

    def on_batch_end(self, batch, logs=None):
        self.epoch_logs[self.epoch].append(logs)


@pytest.mark.parametrize(
    ["num_rows", "batch_size", "train_metrics_steps", "expected_steps", "expected_metrics_steps"],
    [
        (1, 1, 1, 1, 1),
        (60, 10, 2, 6, 3),
        (60, 10, 3, 6, 2),
        (120, 10, 4, 12, 3),
    ],
)
def test_train_metrics_steps(
    num_rows, batch_size, train_metrics_steps, expected_steps, expected_metrics_steps
):
    dataset = generate_data("e-commerce", num_rows=num_rows)
    model = ml.Model(
        ml.InputBlock(dataset.schema),
        ml.MLPBlock([64]),
        ml.BinaryClassificationTask("click"),
    )
    model.compile(
        run_eagerly=True,
        optimizer="adam",
        metrics=[tf.keras.metrics.AUC(from_logits=True, name="auc")],
    )
    metrics_callback = MetricsLogger()
    callbacks = [metrics_callback]
    _ = model.fit(
        dataset,
        callbacks=callbacks,
        epochs=1,
        batch_size=batch_size,
        train_metrics_steps=train_metrics_steps,
    )
    epoch0_logs = metrics_callback.epoch_logs[0]

    # number of times compute_metrics called (number of batches in epoch)
    assert len(epoch0_logs) == expected_steps

    # number of times metrics computed (every train_metrics_steps batches)
    assert len({metrics["auc"] for metrics in epoch0_logs}) == expected_metrics_steps
