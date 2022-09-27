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
import copy

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.test import TestCase

import merlin.models.tf as mm
from merlin.datasets.synthetic import generate_data
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils, tf_utils
from merlin.schema import ColumnSchema, Schema, Tags


@pytest.mark.parametrize("run_eagerly", [False])
def test_simple_model(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([4]),
        mm.BinaryClassificationTask("click"),
    )

    loaded_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    features = ecommerce_data.schema.remove_by_tag(Tags.TARGET).column_names
    testing_utils.test_model_signature(loaded_model, features, ["click/binary_classification_task"])


def test_fit_twice():
    dataset = Dataset(pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "target": [1, 0, 0, 1, 1, 0]}))
    dataset.schema = Schema(
        [
            ColumnSchema("feature", dtype=np.int32, tags=[Tags.CONTINUOUS]),
            ColumnSchema("target", dtype=np.int32, tags=[Tags.BINARY_CLASSIFICATION]),
        ]
    )
    loader = mm.Loader(dataset, batch_size=2, shuffle=False)
    model = mm.Model(
        tf.keras.layers.Lambda(lambda x: x["feature"]),
        tf.keras.layers.Dense(1),
        mm.BinaryClassificationTask("target"),
    )
    model.compile(run_eagerly=True, optimizer="adam")
    model.fit(loader, epochs=2)
    model.fit(loader, epochs=2)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_from_block(ecommerce_data: Dataset, run_eagerly):
    embedding_options = mm.EmbeddingOptions(embedding_dim_default=2)
    model = mm.Model.from_block(
        mm.MLPBlock([4]),
        ecommerce_data.schema,
        prediction_tasks=mm.BinaryClassificationTask("click"),
        embedding_options=embedding_options,
    )

    assert all(
        [f.table.dim == 2 for f in list(model.blocks[0]["categorical"].feature_config.values())]
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


def test_block_from_model_with_input(ecommerce_data: Dataset):
    inputs = mm.InputBlock(ecommerce_data.schema)
    block = inputs.connect(mm.MLPBlock([2]))

    with pytest.raises(ValueError) as excinfo:
        mm.Model.from_block(
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


class UpdateCountMetric(tf.keras.metrics.Metric):
    """Metric that returns a value representing the number of times it has been updated."""

    def __init__(self, name="update_count_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self._built = False

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._built:
            self.call_count = self.add_weight(
                "call_count", shape=tf.TensorShape([1]), initializer="zeros"
            )
            self._built = True

        self.call_count.assign(self.call_count + tf.constant([1.0]))

    def result(self):
        return self.call_count[0]

    def reset_state(self):
        self.call_count.assign(tf.constant([0.0]))


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
    model = mm.Model(
        mm.InputBlock(dataset.schema),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
    )
    model.compile(
        run_eagerly=True,
        optimizer="adam",
        metrics=[UpdateCountMetric()],
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
    assert (
        len({metrics["update_count_metric"] for metrics in epoch0_logs}) == expected_metrics_steps
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_pre_post(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([4]),
        mm.BinaryClassificationTask("click"),
        post=mm.NoOp(),
    )

    model.pre = mm.StochasticSwapNoise(ecommerce_data.schema)

    loaded_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert isinstance(loaded_model.pre, mm.StochasticSwapNoise)
    assert isinstance(loaded_model.post, mm.NoOp)


def test_sub_class_model(ecommerce_data: Dataset):
    blocks = ["input_block", "mlp", "prediction"]

    @tf.keras.utils.register_keras_serializable(package="merlin.models")
    class SubClassedModel(mm.BaseModel):
        def __init__(self, schema: Schema, target: str, **kwargs):
            super(SubClassedModel, self).__init__()
            if "input_block" not in kwargs:
                self.input_block = mm.InputBlock(schema)
                self.mlp = mm.MLPBlock([4])
                self.prediction = mm.BinaryClassificationTask(target)
            else:
                self.input_block = kwargs["input_block"]
                self.mlp = kwargs["mlp"]
                self.prediction = kwargs["prediction"]

        def call(self, inputs, **kwargs):
            x = self.input_block(inputs)
            x = self.mlp(x)

            return self.prediction(x)

        def get_config(self):
            config = {}

            return tf_utils.maybe_serialize_keras_objects(self, config, blocks)

        @classmethod
        def from_config(cls, config):
            config = tf_utils.maybe_deserialize_keras_objects(config, blocks)

            return cls(None, None, **config)

    model = SubClassedModel(ecommerce_data.schema, "click")
    testing_utils.model_test(model, ecommerce_data)


def test_find_blocks_and_sub_blocks(ecommerce_data):
    test_case = TestCase()
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    input_block = mm.InputBlockV2(schema)
    layer_1 = mm.MLPBlock([64], name="layer_1")
    layer_2 = mm.MLPBlock([1], no_activation_last_layer=True, name="layer_2")
    two_layer = mm.SequentialBlock([layer_1, layer_2], name="two_layers")
    body = input_block.connect(two_layer)

    model = mm.Model(body, mm.BinaryClassificationTask("click"))
    testing_utils.model_test(model, ecommerce_data)

    print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
    Model: "model"
    ___________________________________________________________________________________________
    Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    sequential_block_1 (SequentialBloc  multiple                       7289        Y
    k)
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block (SequentialBlock)  multiple                     5624        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || as_ragged_features (AsRaggedFeatur  multiple                   0           Y          ||
    || es)                                                                                   ||
    ||                                                                                       ||
    || parallel_block (ParallelBlock)  multiple                       5624        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| embeddings (ParallelBlock)   multiple                        5624        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| user_categories (EmbeddingTable)  multiple                 4816        Y          ||||
    ||||                                                                                   ||||
    |||| item_category (EmbeddingTable)  multiple                   808         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | two_layers (SequentialBlock)     multiple                        1665        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_1 (SequentialBlock)      multiple                        1600        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense (_Dense)      multiple                        1600        Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_2 (SequentialBlock)      multiple                        65          Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_1 (_Dense)    multiple                        65          Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       2           Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           Y

    ===========================================================================================
    """

    assert model.get_blocks_by_name("layer_1") == [layer_1]
    assert layer_1 in tf_utils.get_sub_blocks(layer_1)
    # Including MLPBlock, _Dense, and Dense
    assert len(tf_utils.get_sub_blocks(layer_1)) == 3

    assert len(model.get_blocks_by_name(["layer_1", "layer_2"])) == 2
    assert model.get_blocks_by_name("two_layers") == [two_layer]
    assert len(tf_utils.get_sub_blocks(two_layer)) == 7
    assert len(model.get_blocks_by_name("user_categories")) == 1

    with test_case.assertRaisesRegex(ValueError, "Cannot find block with the name of"):
        model.get_blocks_by_name("two_layer")


# TODO: make it work for graph mode (run_eagerly = False)
@pytest.mark.parametrize("run_eagerly", [True])
def test_freeze_parallel_block(ecommerce_data, run_eagerly):
    # Train all parameters at first then freeze some layers
    test_case = TestCase()
    schema = ecommerce_data.schema.select_by_name(
        names=["user_categories", "item_category", "click"]
    )
    input_block = mm.InputBlockV2(schema)
    layer_1 = mm.MLPBlock([64], name="layer_1")
    body = input_block.connect(layer_1)

    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    # Compile(Make sure set run_eagerly mode) and fit -> model.freeze_blocks -> compile and fit
    # Set run_eagerly=True in order to avoid error: "Called a function referencing variables which
    # have been deleted". Model needs to be built by fit or build.

    # Currently works well with run_eagerly=True, for graph mode (run_eagerly=False), when
    # re-compile and fit, related to issue #647.
    # TODO: make it work for graph mode (run_eagerly = False), not only for this test, but also for
    # all the tests related to layer-freezing.
    model.compile(run_eagerly=run_eagerly, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    model.freeze_blocks(["user_categories"])
    print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
    Model: "model"
    ___________________________________________________________________________________________
    Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    sequential_block_1 (SequentialBloc  multiple                       7224        Y
    k)
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block (SequentialBlock)  multiple                     5624        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || as_ragged_features (AsRaggedFeatur  multiple                   0           Y          ||
    || es)                                                                                   ||
    ||                                                                                       ||
    || parallel_block (ParallelBlock)  multiple                       5624        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| embeddings (ParallelBlock)   multiple                        5624        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| user_categories (EmbeddingTable)  multiple                 4816        N          ||||
    ||||                                                                                   ||||
    |||| item_category (EmbeddingTable)  multiple                   808         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | layer_1 (SequentialBlock)        multiple                        1600        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || private__dense (_Dense)        multiple                        1600        Y          ||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       65          Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           N

    ===========================================================================================
    """
    assert model.get_blocks_by_name("user_categories")[0].trainable is False

    # Record weights before training
    frozen_weights = {}
    unfrozen_weights = {}

    user_categories_block = model.get_blocks_by_name("user_categories")[0]
    assert len(user_categories_block.trainable_weights) == 0
    assert len(user_categories_block.non_trainable_weights) == 1
    frozen_weights["user_categories"] = copy.deepcopy(user_categories_block.weights[0].numpy())

    item_category_block = model.get_blocks_by_name("item_category")[0]
    assert len(item_category_block.trainable_weights) == 1
    assert len(item_category_block.non_trainable_weights) == 0
    unfrozen_weights["item_category"] = copy.deepcopy(item_category_block.weights[0].numpy())

    layer_1_block = model.get_blocks_by_name("layer_1")[0]
    assert len(layer_1_block.trainable_weights) == 2
    assert len(layer_1_block.non_trainable_weights) == 0
    unfrozen_weights["layer_1"] = copy.deepcopy(layer_1_block.weights[0].numpy())

    # Train model
    # The last time compile and fit, so run_eagerly=False is working
    model.compile(run_eagerly=False, optimizer="adam")
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    # Compare whether weights are updated or not
    test_case.assertAllClose(
        frozen_weights["user_categories"],
        user_categories_block.weights[0].numpy(),
    )
    test_case.assertNotAllClose(unfrozen_weights["layer_1"], layer_1_block.weights[0].numpy())
    test_case.assertNotAllClose(
        unfrozen_weights["item_category"], item_category_block.weights[0].numpy()
    )


def test_freeze_sequential_block(ecommerce_data):
    test_case = TestCase()
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    input_block = mm.InputBlockV2(schema)
    layer_1 = mm.MLPBlock([64], name="layer_1")
    layer_2 = mm.MLPBlock([1], no_activation_last_layer=True, name="layer_2")
    two_layer = mm.SequentialBlock([layer_1, layer_2], name="two_layers")
    body = input_block.connect(two_layer)

    model = mm.Model(body, mm.BinaryClassificationTask("click"))
    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    model.freeze_blocks(["user_categories", "layer_2"])
    print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
    Model: "model"
    ___________________________________________________________________________________________
    Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    sequential_block_1 (SequentialBloc  multiple                       7289        Y
    k)
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block (SequentialBlock)  multiple                     5624        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || as_ragged_features (AsRaggedFeatur  multiple                   0           Y          ||
    || es)                                                                                   ||
    ||                                                                                       ||
    || parallel_block (ParallelBlock)  multiple                       5624        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| embeddings (ParallelBlock)   multiple                        5624        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| user_categories (EmbeddingTable)  multiple                 4816        N          ||||
    ||||                                                                                   ||||
    |||| item_category (EmbeddingTable)  multiple                   808         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | two_layers (SequentialBlock)     multiple                        1665        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_1 (SequentialBlock)      multiple                        1600        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense (_Dense)      multiple                        1600        Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_2 (SequentialBlock)      multiple                        65          N          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_1 (_Dense)    multiple                        65          N          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       2           Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           N

    ===========================================================================================
    """
    assert model.get_blocks_by_name("user_categories")[0].trainable is False
    assert model.get_blocks_by_name("layer_2")[0].trainable is False
    assert all(block.trainable is False for block in tf_utils.get_sub_blocks(layer_2))

    # Record weights before training
    frozen_weights = {}
    unfrozen_weights = {}

    user_categories_block = model.get_blocks_by_name("user_categories")[0]
    assert len(user_categories_block.trainable_weights) == 0
    assert len(user_categories_block.non_trainable_weights) == 1
    frozen_weights["user_categories"] = copy.deepcopy(user_categories_block.weights[0].numpy())

    item_category_block = model.get_blocks_by_name("item_category")[0]
    assert len(item_category_block.trainable_weights) == 1
    assert len(item_category_block.non_trainable_weights) == 0
    unfrozen_weights["item_category"] = copy.deepcopy(item_category_block.weights[0].numpy())

    layer_1_block = model.get_blocks_by_name("layer_1")[0]
    assert len(layer_1_block.trainable_weights) == 2
    assert len(layer_1_block.non_trainable_weights) == 0
    unfrozen_weights["layer_1"] = copy.deepcopy(layer_1.weights[0].numpy())

    layer_2_block = model.get_blocks_by_name("layer_2")[0]
    assert len(layer_2_block.trainable_weights) == 0
    assert len(layer_2_block.non_trainable_weights) == 2
    frozen_weights["layer_2"] = copy.deepcopy(layer_2_block.weights[0].numpy())

    # Train model
    model.compile(run_eagerly=False, optimizer="adam")
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    # Compare whether weights are updated or not
    test_case.assertAllClose(
        frozen_weights["user_categories"],
        user_categories_block.weights[0].numpy(),
    )
    test_case.assertAllClose(frozen_weights["layer_2"], layer_2.weights[0].numpy())
    test_case.assertNotAllClose(unfrozen_weights["layer_1"], layer_1.weights[0].numpy())
    test_case.assertNotAllClose(
        unfrozen_weights["item_category"], item_category_block.weights[0].numpy()
    )


def test_freeze_unfreeze(ecommerce_data):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    input_block = mm.InputBlockV2(schema)
    layer_1 = mm.MLPBlock([64], name="layer_1")
    layer_2 = mm.MLPBlock([1], no_activation_last_layer=True, name="layer_2")
    two_layer = mm.SequentialBlock([layer_1, layer_2], name="two_layers")
    body = input_block.connect(two_layer)

    model = mm.Model(body, mm.BinaryClassificationTask("click"))
    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    model.freeze_blocks(["user_categories", "layer_2"])
    print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
    Model: "model"
    ___________________________________________________________________________________________
    Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    sequential_block_1 (SequentialBloc  multiple                       7289        Y
    k)
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block (SequentialBlock)  multiple                     5624        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || as_ragged_features (AsRaggedFeatur  multiple                   0           Y          ||
    || es)                                                                                   ||
    ||                                                                                       ||
    || parallel_block (ParallelBlock)  multiple                       5624        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| embeddings (ParallelBlock)   multiple                        5624        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| user_categories (EmbeddingTable)  multiple                 4816        N          ||||
    ||||                                                                                   ||||
    |||| item_category (EmbeddingTable)  multiple                   808         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | two_layers (SequentialBlock)     multiple                        1665        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_1 (SequentialBlock)      multiple                        1600        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense (_Dense)      multiple                        1600        Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_2 (SequentialBlock)      multiple                        65          N          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_1 (_Dense)    multiple                        65          N          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       2           Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           N

    ===========================================================================================
    """
    assert len(model.frozen_blocks) == 2
    assert model.get_blocks_by_name("user_categories")[0].trainable is False
    assert len(model.get_blocks_by_name("user_categories")[0].trainable_weights) == 0
    assert model.get_blocks_by_name("layer_2")[0].trainable is False
    assert len(model.get_blocks_by_name("layer_2")[0].trainable_weights) == 0
    assert all(block.trainable is False for block in tf_utils.get_sub_blocks(layer_2))

    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    # Unfreeze
    model.unfreeze_blocks(["user_categories", "layer_2"])

    assert len(model.frozen_blocks) == 0
    assert model.get_blocks_by_name("user_categories")[0].trainable is True
    assert model.get_blocks_by_name("layer_2")[0].trainable is True
    assert all(block.trainable is True for block in tf_utils.get_sub_blocks(layer_2))
    assert len(model.get_blocks_by_name("user_categories")[0].trainable_weights) == 1
    assert len(model.get_blocks_by_name("layer_2")[0].trainable_weights) == 2

    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)


def test_unfreeze_all_blocks(ecommerce_data):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    input_block = mm.InputBlockV2(schema)
    layer_1 = mm.MLPBlock([64], name="layer_1")
    layer_2 = mm.MLPBlock([1], no_activation_last_layer=True, name="layer_2")
    two_layer = mm.SequentialBlock([layer_1, layer_2], name="two_layers")
    body = input_block.connect(two_layer)

    model = mm.Model(body, mm.BinaryClassificationTask("click"))
    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    model.freeze_blocks(["user_categories", "layer_2"])
    print(model.summary(expand_nested=True, show_trainable=True, line_length=80))
    """
    Model: "model"
    ___________________________________________________________________________________________
    Layer (type)                       Output Shape                    Param #     Trainable
    ===========================================================================================
    sequential_block_1 (SequentialBloc  multiple                       7289        Y
    k)
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sequential_block (SequentialBlock)  multiple                     5624        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || as_ragged_features (AsRaggedFeatur  multiple                   0           Y          ||
    || es)                                                                                   ||
    ||                                                                                       ||
    || parallel_block (ParallelBlock)  multiple                       5624        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| embeddings (ParallelBlock)   multiple                        5624        Y          |||
    ||||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||||
    |||| user_categories (EmbeddingTable)  multiple                 4816        N          ||||
    ||||                                                                                   ||||
    |||| item_category (EmbeddingTable)  multiple                   808         Y          ||||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | two_layers (SequentialBlock)     multiple                        1665        Y          |
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_1 (SequentialBlock)      multiple                        1600        Y          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense (_Dense)      multiple                        1600        Y          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    || layer_2 (SequentialBlock)      multiple                        65          N          ||
    |||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|||
    ||| private__dense_1 (_Dense)    multiple                        65          N          |||
    ||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    click/binary_classification_task (  multiple                       2           Y
    BinaryClassificationTask)

    model_context (ModelContext)       multiple                        0           N

    ===========================================================================================
    """
    assert len(model.frozen_blocks) == 2
    assert model.get_blocks_by_name("user_categories")[0].trainable is False
    assert len(model.get_blocks_by_name("user_categories")[0].trainable_weights) == 0
    assert model.get_blocks_by_name("layer_2")[0].trainable is False
    assert len(model.get_blocks_by_name("layer_2")[0].trainable_weights) == 0
    assert all(block.trainable is False for block in tf_utils.get_sub_blocks(layer_2))

    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)

    # Unfreeze
    model.unfreeze_all_frozen_blocks()

    assert len(model.frozen_blocks) == 0
    assert model.get_blocks_by_name("user_categories")[0].trainable is True
    assert model.get_blocks_by_name("layer_2")[0].trainable is True
    assert all(block.trainable is True for block in tf_utils.get_sub_blocks(layer_2))
    assert len(model.get_blocks_by_name("user_categories")[0].trainable_weights) == 1
    assert len(model.get_blocks_by_name("user_categories")[0].non_trainable_weights) == 0
    assert len(model.get_blocks_by_name("layer_2")[0].trainable_weights) == 2
    assert len(model.get_blocks_by_name("layer_2")[0].non_trainable_weights) == 0

    model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.fit(ecommerce_data, batch_size=128, epochs=1)
