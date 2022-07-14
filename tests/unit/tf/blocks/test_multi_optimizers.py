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
from merlin.models.tf.blocks.core.combinators import ParallelBlock, SequentialBlock, TabularBlock
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
                    (optimizers[0], layers[t][0]),
                    (optimizers[1], layers[t][1]),
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
    test_case.assertNotEqual(
        layers["first_opt"][0].trainable_variables[0], layers["multi_opt"][1].trainable_variables[0]
    )
    test_case.assertNotEqual(
        layers["second_opt"][1].trainable_variables[0],
        layers["multi_opt"][0].trainable_variables[0],
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_with_multi_optimizers(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM))
    # user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    # item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect( ml.MLPBlock([256, 128]))
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    print(model)
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            (tf.keras.optimizers.SGD(), user_tower),
            (tf.keras.optimizers.Adam(), item_tower),
        ],
    )

    model.compile(run_eagerly=run_eagerly, optimizer=multi_optimizers)
    model.fit(ecommerce_data, batch_size=50, epochs=2)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_with_multi_optimizers_2(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    item_category_col_schema = schema.select_by_name("item_category").first
    embedding_layer_item = ml.EmbeddingTable(dim=64, col_schema=item_category_col_schema)
    user_genres_col_schema = schema.select_by_name("user_categories").first
    embedding_layer_user = ml.EmbeddingTable(dim=64, col_schema=user_genres_col_schema)
    item_tower = SequentialBlock(
        tf.keras.layers.Lambda(lambda features: features["item_category"]), embedding_layer_item
    )
    user_tower = SequentialBlock(
        tf.keras.layers.Lambda(lambda features: features["user_categories"]), embedding_layer_user
    )
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    print(model)
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            (tf.keras.optimizers.SGD(), user_tower),
            (tf.keras.optimizers.Adam(), item_tower),
        ],
    )

    model.compile(run_eagerly=run_eagerly, optimizer=multi_optimizers)
    model.fit(ecommerce_data, batch_size=50, epochs=2)
