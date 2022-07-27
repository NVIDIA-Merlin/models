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
import tensorflow as tf
from tensorflow.test import TestCase

import merlin.models.tf as ml
from merlin.datasets.synthetic import generate_data
from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock, TabularBlock
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def generate_two_layers():
    initializer_first_layer = tf.constant_initializer(np.ones((3, 4)))
    initializer_second_layer = tf.constant_initializer(np.ones((4, 1)))
    first_layer = ml.MLPBlock(
        [4], use_bias=False, kernel_initializer=initializer_first_layer, block_name="first_mlp"
    )
    second_layer = ml.MLPBlock(
        [1],
        use_bias=False,
        kernel_initializer=initializer_second_layer,
        block_name="second_mlp",
    )
    return first_layer, second_layer


@pytest.mark.parametrize(
    "optimizers",
    [
        ("sgd", "adam"),
        ("rmsprop", "sgd"),
        ("adam", "adagrad"),
        ("adagrad", "rmsprop"),
        (tf.keras.optimizers.SGD(), tf.keras.optimizers.Adagrad()),
    ],
)
def test_optimizers(optimizers):
    testing_data = generate_data("e-commerce", num_rows=5)
    train_schema = testing_data.schema.select_by_name(
        names=["user_categories", "user_brands", "user_shops", "user_intensions"]
    )
    # "first_opt" means set optimizer of the model with the first optimizers (optimizers[0])
    # "multi_opt" means set optimizer with multi_optimizers, first layer with the first optimizer
    # and the second layer with the second optimizer
    test_cases = ["first_opt", "second_opt", "multi_opt"]
    models, layers = {}, {}
    for t in test_cases:
        layers[t] = generate_two_layers()
        input_block = ParallelBlock(TabularBlock.from_schema(schema=train_schema), is_input=True)
        models[t] = ml.Model.from_block(
            block=SequentialBlock([input_block, layers[t][0], layers[t][1]]),
            schema=train_schema,
            prediction_tasks=ml.BinaryClassificationTask("click"),
        )
        if t == "first_opt":
            optimizer = optimizers[0]
        if t == "second_opt":
            optimizer = optimizers[1]
        if t == "multi_opt":
            multi_optimizer = ml.MultiOptimizer(
                default_optimizer=optimizers[0],
                optimizers_and_blocks=[
                    ml.OptimizerBlocks(optimizers[0], layers[t][0]),
                    ml.OptimizerBlocks(optimizers[1], layers[t][1]),
                ],
            )
            optimizer = multi_optimizer
        # run only one batch
        tf.keras.utils.set_random_seed(1)
        models[t].compile(optimizer=optimizer)
        models[t].fit(testing_data, batch_size=5, epochs=1)

    # Compare trainable variables updated by the same optimizer
    test_case = TestCase()
    test_case.assertAllClose(
        layers["first_opt"][0].trainable_variables[0], layers["multi_opt"][0].trainable_variables[0]
    )
    test_case.assertAllClose(
        layers["second_opt"][1].trainable_variables[0],
        layers["multi_opt"][1].trainable_variables[0],
    )

    # Trainable variables updated by different optimizer are not equal
    test_case.assertNotAllEqual(
        layers["first_opt"][0].trainable_variables[0], layers["multi_opt"][1].trainable_variables[0]
    )
    test_case.assertNotAllEqual(
        layers["second_opt"][1].trainable_variables[0],
        layers["multi_opt"][0].trainable_variables[0],
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_with_multi_optimizers(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            ml.OptimizerBlocks(tf.keras.optimizers.SGD(), user_tower),
            ml.OptimizerBlocks(tf.keras.optimizers.Adam(), item_tower),
        ],
    )
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizers
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_multi_optimizer_list_input(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([128, 64]))
    two_tower = ml.ParallelBlock(
        {"user": user_tower, "item": item_tower, "third": third_tower}, aggregation="concat"
    )
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    multi_optimizers = ml.MultiOptimizer(
        optimizers_and_blocks=[
            ml.OptimizerBlocks(tf.keras.optimizers.SGD(), user_tower),
            ml.OptimizerBlocks(tf.keras.optimizers.Adam(), [item_tower, third_tower]),
        ],
    )
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizers
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_multi_optimizer_add(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([128, 64]))
    two_tower = ml.ParallelBlock(
        {"user": user_tower, "item": item_tower, "third": third_tower}, aggregation="concat"
    )
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            ml.OptimizerBlocks(tf.keras.optimizers.SGD(), user_tower),
            ml.OptimizerBlocks(tf.keras.optimizers.Adam(), item_tower),
        ],
    )
    multi_optimizers.add(ml.OptimizerBlocks("adagrad", third_tower))
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizers
    )


@pytest.mark.parametrize(
    "optimizers",
    [
        ("sgd", "adam"),
        ("rmsprop", "sgd"),
        ("adam", "adagrad"),
        ("adagrad", "rmsprop"),
        (tf.keras.optimizers.SGD(), tf.keras.optimizers.Adagrad()),
    ],
)
def test_multi_optimizers_from_config(ecommerce_data, optimizers):
    test_case = TestCase()
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            ml.OptimizerBlocks(optimizers[0], user_tower),
            ml.OptimizerBlocks(optimizers[1], item_tower),
        ],
    )
    cloned_multi_optimizers = ml.MultiOptimizer.from_config(multi_optimizers.get_config())
    for i in range(len(multi_optimizers.optimizers_and_blocks)):
        optimizer = multi_optimizers.optimizers_and_blocks[i].optimizer
        cloned_optimizer = cloned_multi_optimizers.optimizers_and_blocks[i].optimizer
        test_case.assertDictEqual(cloned_optimizer.get_config(), optimizer.get_config())
    test_case.assertDictEqual(
        cloned_multi_optimizers.default_optimizer.get_config(),
        multi_optimizers.default_optimizer.get_config(),
    )


@pytest.mark.parametrize(
    "optimizers",
    [
        ("sgd", "adam"),
        (tf.keras.optimizers.SGD(), tf.keras.optimizers.Adagrad()),
    ],
)
def test_multi_optimizers_from_config_list_input(ecommerce_data, optimizers):
    test_case = TestCase()
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([128, 64]))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            ml.OptimizerBlocks(optimizers[0], [user_tower, third_tower]),
            ml.OptimizerBlocks(optimizers[1], item_tower),
        ],
    )
    cloned_multi_optimizers = ml.MultiOptimizer.from_config(multi_optimizers.get_config())
    for i in range(len(multi_optimizers.optimizers_and_blocks)):
        optimizer = multi_optimizers.optimizers_and_blocks[i].optimizer
        cloned_optimizer = cloned_multi_optimizers.optimizers_and_blocks[i].optimizer
        test_case.assertDictEqual(cloned_optimizer.get_config(), optimizer.get_config())
    test_case.assertDictEqual(
        cloned_multi_optimizers.default_optimizer.get_config(),
        multi_optimizers.default_optimizer.get_config(),
    )


@pytest.mark.parametrize("use_default", [True, False])
def test_examples_in_code_comments(ecommerce_data, use_default):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([512, 256]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([512, 256]))
    third_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([64]))
    three_tower = ml.ParallelBlock(
        {"user": user_tower, "item": item_tower, "third": third_tower}, aggregation="concat"
    )
    # model = ml.Model(three_tower, ml.ItemRetrievalTask())
    model = ml.Model(three_tower, ml.BinaryClassificationTask("click"))

    # The third_tower would be assigned the default_optimizer ("adagrad" in this example)
    if use_default:
        optimizer = ml.MultiOptimizer(
            default_optimizer="adagrad",
            optimizers_and_blocks=[
                ml.OptimizerBlocks(tf.keras.optimizers.SGD(), user_tower),
                ml.OptimizerBlocks(tf.keras.optimizers.Adam(), item_tower),
            ],
        )
    else:
        # The string identification of optimizer is also acceptable, here "sgd" for the third_tower
        # the variables of BinaryClassificationTask("click") would still use the default_optimizer
        optimizer = ml.MultiOptimizer(
            default_optimizer="adam",
            optimizers_and_blocks=[
                ml.OptimizerBlocks("sgd", [user_tower, third_tower]),
                ml.OptimizerBlocks("adam", item_tower),
            ],
        )

    model.compile(optimizer=optimizer)
    model.fit(ecommerce_data, batch_size=32, epochs=1)
    optimizer.weights
    optimizer.variables
    assert len(optimizer.optimizers) == 3
