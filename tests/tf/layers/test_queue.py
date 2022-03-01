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

import merlin.models.tf as ml

tf = pytest.importorskip("tensorflow")


@pytest.fixture
def fifo_queue_fixture():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[5],
        dtype=tf.float32,
    )

    return queue


def test_queue_single_enqueue_dequeue(fifo_queue_fixture):
    queue = fifo_queue_fixture

    input = tf.random.uniform((5,))
    queue.enqueue(input)

    output = queue.dequeue()
    assert tf.shape(output)[0] == 5
    assert tf.reduce_all(input == output)


def test_queue_multiple_single_enqueue_dequeue(fifo_queue_fixture):
    queue = fifo_queue_fixture

    inputs = [tf.random.uniform((5,)) for _ in range(4)]
    for input in inputs:
        queue.enqueue(input)

    for input in inputs:
        output = queue.dequeue()
        assert tf.reduce_all(input == output)


def test_queue_dequeue_error_when_fully_emptied(fifo_queue_fixture):
    queue = fifo_queue_fixture

    input = tf.random.uniform((5,))
    queue.enqueue(input)

    _ = queue.dequeue()

    with pytest.raises(IndexError) as excinfo:
        _ = queue.dequeue()
    assert "The queue is empty" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        _ = queue.dequeue_many(3)
    assert "The queue is empty" in str(excinfo.value)


def test_queue_dequeue_error_when_nothing_added(fifo_queue_fixture):
    queue = fifo_queue_fixture

    with pytest.raises(IndexError) as excinfo:
        _ = queue.dequeue()
    assert "The queue is empty" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        _ = queue.dequeue_many(3)
    assert "The queue is empty" in str(excinfo.value)


def test_queue_enqueue_dequeue_many(fifo_queue_fixture):
    queue = fifo_queue_fixture

    input = tf.random.uniform((3, 5))
    queue.enqueue_many(input)

    output = queue.dequeue_many(3)
    assert tf.shape(output)[0] == 3
    assert tf.shape(output)[1] == 5
    assert tf.reduce_all(input == output)


def test_queue_enqueue_last_dequeue_many_smaller(fifo_queue_fixture):
    queue = fifo_queue_fixture

    outputs_current = queue.list_all()
    assert tf.shape(outputs_current)[0] == 0

    input = tf.random.uniform((3, 5))
    queue.enqueue_many(input)

    outputs_current = queue.list_all()
    assert tf.reduce_all(outputs_current == input)

    output = queue.dequeue_many(2)
    assert tf.shape(output)[0] == 2
    assert tf.reduce_all(output == input[:2])

    outputs_current = queue.list_all()
    assert tf.reduce_all(outputs_current == input[2:])

    output = queue.dequeue_many(2)
    assert tf.shape(output)[0] == 1
    assert tf.reduce_all(output == input[2:])

    outputs_current = queue.list_all()
    assert tf.shape(outputs_current)[0] == 0


def test_queue_enqueue_until_exceeds_capacity(fifo_queue_fixture):
    queue = fifo_queue_fixture

    inputs = list([tf.random.uniform((3, 5)) for _ in range(4)])
    for input in inputs:
        queue.enqueue_many(input)

    outputs_list = queue.list_all()
    # After adding 12 elements to a 10-sized queue, should keep the last 10 added
    assert tf.reduce_all(outputs_list == tf.concat(inputs, axis=0)[2:])


def test_queue_enqueue_dequeue_consistent_size_and_list(fifo_queue_fixture):
    queue = fifo_queue_fixture

    for i in range(1, 12):
        queue.enqueue(tf.random.uniform((5,)))
        if i < queue.capacity:
            assert queue.count() == i
            assert queue.list_all().shape[0] == i
            assert not queue.at_full_capacity
        else:
            assert queue.count() == queue.capacity
            assert queue.list_all().shape[0] == queue.capacity
            assert queue.at_full_capacity

    for i in range(9, 0):
        _ = queue.dequeue()
        assert queue.count() == i
        assert queue.list_all().shape[0] == i
        assert not queue.at_full_capacity


def test_queue_clear(fifo_queue_fixture):
    queue = fifo_queue_fixture

    for i in range(1, 12):
        queue.enqueue(tf.random.uniform((5,)))

    assert queue.count() == queue.capacity
    assert queue.list_all().shape[0] == queue.capacity
    assert queue.at_full_capacity

    queue.clear()
    assert queue.count() == 0
    assert queue.list_all().shape[0] == 0
    assert not queue.at_full_capacity


def test_enqueue_tensors_1_dim_int32():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[],
        dtype=tf.int32,
    )

    single_input = tf.random.uniform((), minval=1, maxval=10000, dtype=tf.int32)
    queue.enqueue(single_input)

    multiple_inputs = tf.random.uniform((7,), minval=1, maxval=10000, dtype=tf.int32)
    queue.enqueue_many(multiple_inputs)


def test_enqueue_tensors_wrong_dim():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[13],
        dtype=tf.float32,
    )

    single_input = tf.random.uniform((12,))
    with pytest.raises(AssertionError) as excinfo:
        queue.enqueue(single_input)
    assert "The shape of val and self.dims should match" in str(excinfo.value)

    multiple_inputs = tf.random.uniform((7, 11))
    with pytest.raises(AssertionError) as excinfo:
        queue.enqueue_many(multiple_inputs)
    assert (
        "The shape of values (ignoring the first dim which is the number of examples) "
        "and self.dims should match"
    ) in str(excinfo.value)


def test_indexof():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[],
        dtype=tf.int32,
    )

    item_ids = tf.range(10, 0, -1, dtype=tf.int32)
    queue.enqueue_many(item_ids)

    indices = queue.index_of([0, 1, 2])
    tf.assert_equal(tf.cast(indices, tf.int32), [-1, 9, 8])


def test_get_values_by_indices():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[],
        dtype=tf.int32,
    )

    item_ids = tf.range(10, 0, -1, dtype=tf.int32)
    queue.enqueue_many(item_ids)

    values = queue.get_values_by_indices([1, 3])
    tf.assert_equal(values, [9, 7])


def test_update_by_indices():
    queue = ml.FIFOQueue(
        capacity=10,
        dims=[],
        dtype=tf.int32,
    )

    item_ids = tf.range(10, 0, -1, dtype=tf.int32)
    queue.enqueue_many(item_ids)

    queue.update_by_indices(indices=tf.constant([[1], [2]]), values=tf.constant([20, 21]))
    values = queue.list_all()
    tf.assert_equal(values, [10, 20, 21, 7, 6, 5, 4, 3, 2, 1])
