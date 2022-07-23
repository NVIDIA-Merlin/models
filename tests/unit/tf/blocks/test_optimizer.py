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
from merlin.core.dispatch import HAS_GPU
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
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            (tf.keras.optimizers.SGD(), user_tower),
            (tf.keras.optimizers.Adam(), item_tower),
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
            (tf.keras.optimizers.SGD(), user_tower),
            (tf.keras.optimizers.Adam(), [item_tower, third_tower]),
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
            (tf.keras.optimizers.SGD(), user_tower),
            (tf.keras.optimizers.Adam(), item_tower),
        ],
    )
    multi_optimizers.add("adagrad", third_tower)
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
            (optimizers[0], user_tower),
            (optimizers[1], item_tower),
        ],
    )
    cloned_multi_optimizers = ml.MultiOptimizer.from_config(multi_optimizers.get_config())
    for i in range(len(multi_optimizers.optimizers_and_blocks)):
        optimizer, block = multi_optimizers.optimizers_and_blocks[i]
        cloned_optimizer, cloned_block = cloned_multi_optimizers.optimizers_and_blocks[i]
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
            (optimizers[0], [user_tower, third_tower]),
            (optimizers[1], item_tower),
        ],
    )
    cloned_multi_optimizers = ml.MultiOptimizer.from_config(multi_optimizers.get_config())
    for i in range(len(multi_optimizers.optimizers_and_blocks)):
        optimizer, blocks = multi_optimizers.optimizers_and_blocks[i]
        cloned_optimizer, cloned_blocks = cloned_multi_optimizers.optimizers_and_blocks[i]
        test_case.assertDictEqual(cloned_optimizer.get_config(), optimizer.get_config())
    test_case.assertDictEqual(
        cloned_multi_optimizers.default_optimizer.get_config(),
        multi_optimizers.default_optimizer.get_config(),
    )


def adam_update_numpy(param, g_t, t, m, v, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    lr_t = lr * np.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - lr_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_lazy_adam_sparse(dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0_np_indices = np.array([0, 2], dtype=np.int32)
    grads0 = tf.IndexedSlices(
        tf.constant(grads0_np[grads0_np_indices]),
        tf.constant(grads0_np_indices),
        tf.constant([3]),
    )
    grads1_np_indices = np.array([0, 2], dtype=np.int32)
    grads1 = tf.IndexedSlices(
        tf.constant(grads1_np[grads1_np_indices]),
        tf.constant(grads1_np_indices),
        tf.constant([3]),
    )
    opt = ml.LazyAdam()

    # Fetch params to validate initial values
    np.testing.assert_allclose([1.0, 1.0, 2.0], var0.numpy(), 1e-6, 1e-6)
    np.testing.assert_allclose([3.0, 3.0, 4.0], var1.numpy(), 1e-6, 1e-6)

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        testing_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        testing_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        testing_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        testing_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64])
def test_lazy_adam_sparse_device_placement(dtype):

    # If a GPU is available, tests that all optimizer ops can be placed on it (i.e. they have GPU
    # kernels).
    if HAS_GPU:
        with tf.device("/GPU:0"):
            var = tf.Variable([[1.0], [2.0]])
            indices = tf.constant([0, 1], dtype=dtype)

            def g_sum():
                return tf.math.reduce_sum(tf.gather(var, indices))

            optimizer = ml.LazyAdam(3.0)
            optimizer.minimize(g_sum, var_list=[var])

    with tf.device("/CPU:0"):
        var_cpu = tf.Variable([[1.0], [2.0]])
        indices_cpu = tf.constant([0, 1], dtype=dtype)

        def g_sum():
            return tf.math.reduce_sum(tf.gather(var_cpu, indices_cpu))

        optimizer = ml.LazyAdam(3.0)
        optimizer.minimize(g_sum, var_list=[var_cpu])


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_lazy_adam_sparse_repeated_indices(dtype):
    # todo: remove the with tf.device once the placement on cpu is enforced.
    with tf.device("CPU:0"):
        repeated_index_update_var = tf.Variable([[1], [2]], dtype=dtype)
        aggregated_update_var = tf.Variable([[1], [2]], dtype=dtype)
        grad_repeated_index = tf.IndexedSlices(
            tf.constant([0.1, 0.1], shape=[2, 1], dtype=dtype),
            tf.constant([1, 1]),
            tf.constant([2, 1]),
        )
        grad_aggregated = tf.IndexedSlices(
            tf.constant([0.2], shape=[1, 1], dtype=dtype),
            tf.constant([1]),
            tf.constant([2, 1]),
        )
        repeated_update_opt = ml.LazyAdam()
        aggregated_update_opt = ml.LazyAdam()
        for _ in range(3):
            repeated_update_opt.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
            aggregated_update_opt.apply_gradients([(grad_aggregated, aggregated_update_var)])
            np.testing.assert_allclose(
                aggregated_update_var.numpy(), repeated_index_update_var.numpy()
            )


@pytest.mark.parametrize("use_callable_params", [True, False])
@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_lazy_adam_callable_lr(use_callable_params, dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)

    def learning_rate():
        return 0.001

    if not use_callable_params:
        learning_rate = learning_rate()

    opt = ml.LazyAdam(learning_rate=learning_rate)

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        testing_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        testing_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        testing_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        testing_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_lazy_adam_tensor_learning_rate(dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = ml.LazyAdam(tf.constant(0.001))

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        testing_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        testing_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        testing_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        testing_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_lazy_adam_sharing(dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = ml.LazyAdam()

    # Fetch params to validate initial values
    np.testing.assert_allclose([1.0, 2.0], var0.numpy())
    np.testing.assert_allclose([3.0, 4.0], var1.numpy())

    # Run 3 steps of intertwined Adam1 and Adam2.
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        testing_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        testing_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        testing_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        testing_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_lazy_adam_slots_unique_eager():
    v1 = tf.Variable(1.0)
    v2 = tf.Variable(1.0)
    opt = ml.LazyAdam(1.0)
    opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
    # There should be iteration, and two unique slot variables for v1 and v2.
    assert 5 == len(opt.variables())
    assert opt.variables()[0] == opt.iterations


def test_lazy_adam_serialization():
    optimizer = ml.LazyAdam()
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_lazy_adam_in_model(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    lazy_adam_optimizer = ml.LazyAdam()
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=lazy_adam_optimizer
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_lazy_adam_in_model_with_multi_optimizers(ecommerce_data, run_eagerly):
    schema = ecommerce_data.schema
    user_tower = ml.InputBlock(schema.select_by_tag(Tags.USER)).connect(ml.MLPBlock([256, 128]))
    item_tower = ml.InputBlock(schema.select_by_tag(Tags.ITEM)).connect(ml.MLPBlock([256, 128]))
    two_tower = ml.ParallelBlock({"user": user_tower, "item": item_tower}, aggregation="concat")
    model = ml.Model(two_tower, ml.BinaryClassificationTask("click"))
    multi_optimizers = ml.MultiOptimizer(
        default_optimizer="adam",
        optimizers_and_blocks=[
            (tf.keras.optimizers.SGD(), user_tower),
            (ml.LazyAdam(), item_tower),
        ],
    )
    testing_utils.model_test(
        model, ecommerce_data, run_eagerly=run_eagerly, optimizer=multi_optimizers
    )
