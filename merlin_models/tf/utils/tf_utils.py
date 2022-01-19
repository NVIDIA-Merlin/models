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
from typing import List, Union

import tensorflow as tf
from tensorflow.python.layers.base import Layer

from merlin_models.tf.typing import TabularData


def get_output_sizes_from_schema(schema, batch_size=0, max_sequence_length=None):
    sizes = {}
    for feature in schema.feature:
        name = feature.name
        if feature.HasField("value_count"):
            sizes[name] = tf.TensorShape(
                [
                    batch_size,
                    max_sequence_length if max_sequence_length else feature.value_count.max,
                ]
            )
        elif feature.HasField("shape"):
            sizes[name] = tf.TensorShape([batch_size] + [d.size for d in feature.shape.dim])
        else:
            sizes[name] = tf.TensorShape([batch_size, 1])

    return sizes


def calculate_batch_size_from_input_shapes(input_shapes):
    return [i for i in input_shapes.values() if not isinstance(i, tuple)][0][0]


def maybe_serialize_keras_objects(
    self,
    config,
    maybe_serialize_keys,
):
    for key in maybe_serialize_keys:
        maybe_value = getattr(self, key, None)
        if maybe_value:
            if isinstance(maybe_value, dict):
                config[key] = {
                    k: tf.keras.utils.serialize_keras_object(v) for k, v in maybe_value.items()
                }
            elif isinstance(maybe_value, (list, tuple)):
                config[key] = [tf.keras.utils.serialize_keras_object(v) for v in maybe_value]
            else:
                config[key] = tf.keras.utils.serialize_keras_object(maybe_value)

    return config


def maybe_deserialize_keras_objects(
    config, to_deserialize, deserialize_fn=tf.keras.utils.deserialize_keras_object
):
    if isinstance(to_deserialize, list):
        to_deserialize = {k: deserialize_fn for k in to_deserialize}

    custom_objects = {}

    for key, fn in to_deserialize.items():
        maybe_val = config.get(key, None)
        if maybe_val:
            if isinstance(maybe_val, list):
                config[key] = [fn(v, custom_objects=custom_objects) for v in maybe_val]
            else:
                config[key] = fn(maybe_val, custom_objects=custom_objects)

    return config


def extract_topk(ks, scores, labels):
    max_k = tf.reduce_max(ks)
    topk_scores, topk_indices = tf.math.top_k(scores, max_k)
    topk_labels = gather_torch_like(labels, topk_indices, max_k)
    return topk_scores, topk_indices, topk_labels


def tranform_label_to_onehot(labels, vocab_size):
    return tf.one_hot(tf.reshape(labels, (-1,)), vocab_size)


def create_output_placeholder(scores, ks):
    return tf.Variable(tf.zeros([tf.shape(scores)[0], len(ks)], tf.float32))


def gather_torch_like(labels, indices, max_k):
    # gather_indices = []
    gather_indices = tf.TensorArray(tf.int32, size=tf.shape(indices)[0])
    for i in range(tf.shape(indices)[0]):
        gather_indices = gather_indices.write(
            i,
            tf.concat(
                [i * tf.ones((max_k, 1), tf.int32), tf.expand_dims(indices[i, :], -1)], axis=1
            ),
        )
    all_indices = gather_indices.stack()
    labels = tf.reshape(tf.gather_nd(labels, all_indices), tf.shape(indices))
    return labels


def batch_ref(inputs: Union[tf.Tensor, TabularData]):
    """Get hash-code of a tensor or a dictionary of tensors."""

    if isinstance(inputs, tf.Tensor):
        return hash(inputs.ref())

    refs = []
    keys = sorted(inputs.keys())
    for key in keys:
        refs.append(inputs[key].ref())

    return hash(tuple(refs))


class FIFOQueue(Layer):
    def __init__(
        self,
        capacity: int,
        dtype: tf.DType,
        dims: List[int] = [],
        queue_name: str = "",
        initialize_tensor: tf.Tensor = None,
        **kwargs,
    ):
        super(FIFOQueue, self).__init__(**kwargs)
        self.capacity = capacity
        self.queue_dtype = dtype
        self.dims = dims
        self.queue_name = queue_name
        self.initialize_tensor = initialize_tensor

        self.first_pointer = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.int32)),
            trainable=False,
            dtype=tf.int32,
        )
        self.next_available_pointer = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.int32)),
            trainable=False,
            dtype=tf.int32,
        )
        self.at_full_capacity = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.bool)),
            trainable=False,
            dtype=tf.bool,
        )

        if initialize_tensor is None:
            initialize_tensor = tf.Variable(lambda: tf.zeros([capacity] + self.dims, dtype=dtype))

        self.storage = tf.Variable(
            initial_value=initialize_tensor,
            name=f"{self.queue_name}/fifo_queue_storage",
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape([self.capacity] + self.dims),
            dtype=self.queue_dtype,
        )

    def enqueue(self, val: tf.Tensor):
        assert len(val.shape) == len(self.dims)
        assert list(val.shape) == self.dims

        self.storage[self.next_available_pointer].assign(val)

        self.next_available_pointer.assign_add(1)
        if self.next_available_pointer >= self.capacity:
            self.next_available_pointer.assign(0)

        if self.at_full_capacity or self.next_available_pointer == self.first_pointer:
            self.first_pointer.assign(self.next_available_pointer)
            self.at_full_capacity.assign(True)

    def enqueue_many(self, vals: tf.Tensor):
        assert len(tf.shape(vals)) == len(self.dims) + 1
        assert list(vals.shape[1:]) == self.dims

        # if values are larger than the queue capacity N, enqueueing only the last N items
        vals = vals[-self.capacity :]
        num_vals = int(tf.shape(vals)[0])

        next_pos_start = self.next_available_pointer
        next_pos_end = next_pos_start + num_vals

        if next_pos_end < self.capacity:

            self.storage[next_pos_start:next_pos_end].assign(vals)

            if self.at_full_capacity or (
                next_pos_start < self.first_pointer and next_pos_end >= self.first_pointer
            ):
                self.first_pointer.assign(next_pos_end)
                self.at_full_capacity.assign(True)
        else:
            num_overplus_items = next_pos_end - self.capacity
            next_pos_end = self.capacity
            self.storage[next_pos_start:next_pos_end].assign(vals[: num_vals - num_overplus_items])

            next_pos_start = 0
            next_pos_end = num_overplus_items
            self.storage[next_pos_start:next_pos_end].assign(vals[num_vals - num_overplus_items :])

            if self.at_full_capacity or next_pos_end >= self.first_pointer:
                self.first_pointer.assign(next_pos_end)
                self.at_full_capacity.assign(True)

        self.next_available_pointer.assign(next_pos_end)

    def dequeue(self):
        if self.first_pointer == self.next_available_pointer:
            raise IndexError("The queue is empty")
        self.at_full_capacity.assign(False)
        val = self.storage[self.first_pointer]
        self.first_pointer.assign_add(1)
        if self.first_pointer >= self.capacity:
            self.first_pointer.assign(0)
        return val

    def dequeue_many(self, n: int):
        if self.first_pointer == self.next_available_pointer:
            raise IndexError("The queue is empty")
        if n <= 0:
            raise ValueError("The number of elements to dequeue must be greater than 0.")
        self.at_full_capacity.assign(False)
        next_pos_start = self.first_pointer
        next_pos_end = next_pos_start + n

        if self.next_available_pointer > self.first_pointer:
            next_pos_end = min(next_pos_end, self.next_available_pointer)

        if next_pos_end < self.capacity:
            vals = self.storage[next_pos_start:next_pos_end]
        else:
            num_missing_items = next_pos_end - self.capacity
            next_pos_end = self.capacity
            vals1 = self.storage[next_pos_start:next_pos_end]

            next_pos_start = 0
            next_pos_end = min(num_missing_items, self.next_available_pointer)
            vals2 = self.storage[next_pos_start:next_pos_end]

            vals = tf.concat([vals1, vals2], axis=0)

        self.first_pointer.assign(next_pos_end)
        return vals

    def list_all(self):
        if self.first_pointer < self.next_available_pointer:
            return self.storage[self.first_pointer : self.next_available_pointer]
        elif self.first_pointer == self.next_available_pointer and not self.at_full_capacity:
            return self.storage[0:0]  # Returns empty Tensor when queue is empty
        else:
            return tf.concat(
                [self.storage[self.first_pointer :], self.storage[: self.next_available_pointer]],
                axis=0,
            )

    def count(self):
        if self.first_pointer < self.next_available_pointer:
            return self.next_available_pointer - self.first_pointer
        elif self.at_full_capacity:
            return self.capacity
        elif self.first_pointer == self.next_available_pointer:
            return 0
        else:
            return self.capacity - self.first_pointer + self.next_available_pointer

    def clear(self):
        self.first_pointer.assign(0)
        self.next_available_pointer.assign(0)
        self.at_full_capacity.assign(False)
