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

from typing import List

import tensorflow as tf
from tensorflow.python.layers.base import Layer


class FIFOQueue(Layer):
    """Fixed size FIFO queue for caching input tensors
    over the batches. The storage is a fixed size tf.Variable()
    and some pointers are used to emulate the `queue()` / `dequeue()`
    ops. As soon as maximum capacity is reached, the oldest examples
    are removed from the queue to add the current beach ones.
    As an example use case, the `FIFOQueue` is used by
    `CachedCrossBatchSampler` to cache item embeddings and item metadata features.

        Parameters
        ----------
        capacity : int
            Maximum number of examples to store
        dtype : tf.DType
            The dtype of tensor that will be stored (e.g. tf.int32, tf.float32)
        dims : List[int], optional
            The dimension of the tensor examples, by default [], which means each example
            is a scalar.
        queue_name : str, optional
            This is string is concat with the tf.Variable() name to allow differentiating in the
            graph the FIFO Queue storages, by default ""
        initialize_tensor : tf.Tensor, optional
            Allows for initializing the storage with values, by default None which initializes
            the storage with -1.
            P.s. It is important that for categorical features the storage Variable is not
            initialized with a valid categorical value (e.g. values >= 0), so that `index_of()`
            works properly
    """

    def __init__(
        self,
        capacity: int,
        dtype: tf.DType,
        dims: List[int] = [],
        queue_name: str = "",
        initialize_tensor: tf.Tensor = None,
        **kwargs,
    ):
        assert capacity > 0

        super(FIFOQueue, self).__init__(**kwargs)
        self.capacity = capacity
        self.queue_dtype = dtype
        self.dims = dims
        self.queue_name = queue_name
        self.initialize_tensor = initialize_tensor

        self.first_pointer = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.int32)),
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            dtype=tf.int32,
        )
        self.next_available_pointer = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.int32)),
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            dtype=tf.int32,
        )
        self.at_full_capacity = tf.Variable(
            initial_value=tf.Variable(lambda: tf.zeros((), dtype=tf.bool)),
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            dtype=tf.bool,
        )

        if initialize_tensor is None:
            initialize_tensor = tf.Variable(
                lambda: tf.zeros([capacity] + self.dims, dtype=dtype) - 1
            )

        self.storage = tf.Variable(
            initial_value=initialize_tensor,
            name=f"{self.queue_name}/fifo_queue_storage",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            shape=tf.TensorShape([self.capacity] + self.dims),
            dtype=self.queue_dtype,
        )

    def enqueue(self, val: tf.Tensor) -> None:
        """Enqueues an example into the queue

        Parameters
        ----------
        val : tf.Tensor
            Tensor to be stored in the queue. Its shape should match the `dims`
            set in the queue constructor
        """
        assert len(val.shape) == len(self.dims), "The rank of val and self.dims should match"
        assert list(val.shape) == self.dims, "The shape of val and self.dims should match"

        self.storage[self.next_available_pointer].assign(val)

        self.next_available_pointer.assign_add(1)
        if self.next_available_pointer >= self.capacity:
            self.next_available_pointer.assign(0)

        if self.at_full_capacity or self.next_available_pointer == self.first_pointer:
            self.first_pointer.assign(self.next_available_pointer)
            self.at_full_capacity.assign(True)

    def _check_input_values(self, values):
        if values.shape.dims is not None:
            assert len(tf.shape(values)) == len(self.dims) + 1, (
                "The rank of values (ignoring the first dim which is the number of examples) and "
                "self.dims should match"
            )
            assert list(values.shape[1:]) == self.dims, (
                "The shape of values (ignoring the first dim which is the number of examples) and "
                "self.dims should match"
            )

    def enqueue_many(self, vals: tf.Tensor) -> None:
        """Enqueues many examples into the queue.

        Parameters
        ----------
        val : tf.Tensor
            Tensor with the examples to be stored in the queue.
            The first dim of `vals` is the number of examples.
            From the second dim, its shape should match the `dims`
            set in the queue constructor
        """
        self._check_input_values(vals)

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

    def dequeue(self) -> tf.Tensor:
        """Dequeues a single example from the queue

        Returns
        -------
        tf.Tensor
            A single example of the queue

        Raises
        ------
        IndexError
            The queue is empty
        """
        if self.first_pointer == self.next_available_pointer:
            raise IndexError("The queue is empty")
        self.at_full_capacity.assign(False)
        val = self.storage[self.first_pointer]
        self.first_pointer.assign_add(1)
        if self.first_pointer >= self.capacity:
            self.first_pointer.assign(0)
        return val

    def dequeue_many(self, n: int) -> tf.Tensor:
        """Dequeues many examples from the queue

        Parameters
        ----------
        n : int
            Number of examples to sample from the queue

        Returns
        -------
        tf.Tensor
            A tensor with N examples, being the first dim equal to N

        Raises
        ------
        IndexError
            The queue is empty
        ValueError
            The number of elements to dequeue must be greater than 0
        """
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

    def list_all(self) -> tf.Tensor:
        """Returns all items in the queue, sorted by the
        order they were added (FIFO)

        Returns
        -------
        tf.Tensor
            Returns a tensor with all examples added to the queue
        """
        if self.first_pointer < self.next_available_pointer:
            return self.storage[self.first_pointer : self.next_available_pointer]
        elif self.first_pointer == self.next_available_pointer and not self.at_full_capacity:
            return self.storage[0:0]  # Returns empty Tensor when queue is empty
        else:
            return tf.concat(
                [self.storage[self.first_pointer :], self.storage[: self.next_available_pointer]],
                axis=0,
            )

    def count(self) -> int:
        """Returns the number of examples added to the queue

        Returns
        -------
        int
            The number of examples added to the queue
        """
        if self.first_pointer < self.next_available_pointer:
            return self.next_available_pointer - self.first_pointer
        elif self.at_full_capacity:
            return self.capacity
        elif self.first_pointer == self.next_available_pointer:
            return 0
        else:
            return self.capacity - self.first_pointer + self.next_available_pointer

    def clear(self) -> None:
        """Removes all examples from the queue"""
        self.first_pointer.assign(0)
        self.next_available_pointer.assign(0)
        self.at_full_capacity.assign(False)

    def index_of(self, ids: tf.Tensor) -> tf.Tensor:
        """Retrieves the indices of the input ids if they exist in the queue,
        and -1 indicates the id was not found.

        Parameters
        ----------
        ids : tf.Tensor
            1D tensor with the search ids

        Returns
        -------
        tf.Tensor
            1D tensor with the same size of the input ids, containing the indices of the
            ids in the queue (-1 if not found)
        """
        assert self.queue_dtype in [tf.int8, tf.int16, tf.int32, tf.int64], (
            "The index_of method is only available for queues with an int dtype "
            "(tf.int8, tf.int16, tf.int32, tf.int64)"
        )
        assert (
            self.dims == []
        ), "The index_of method is only available for queues of scalars (dims=[])"

        # item_ids_indices = tf.where(tf.equal(ids, self.storage))
        equal_tensor = tf.cast(tf.equal(self.storage, tf.expand_dims(ids, -1)), tf.int32)
        # Artificially adding a first zero column so that index 0 is returned if
        # there is not an equal value (1) row-wise
        equal_tensor_extended = tf.concat(
            [tf.zeros(shape=(tf.shape(equal_tensor)[0], 1), dtype=tf.int32), equal_tensor],
            axis=1,
        )
        # Subtracting the indices to account for the first column added, so that
        # values not found become -1
        item_ids_indices = tf.argmax(equal_tensor_extended, axis=1) - 1
        return item_ids_indices

    def get_values_by_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """Retrieves values of the queue based on their index

        Parameters
        ----------
        indices : tf.Tensor
            Indices to retrieve

        Returns
        -------
        tf.Tensor
            Values corresponding to the indices
        """
        result = tf.gather(self.storage, indices)
        return result

    def update_by_indices(self, indices: tf.Tensor, values: tf.Tensor) -> None:
        """Update values of the queue for specific indices

        Parameters
        ----------
        indices : tf.Tensor
            Indices to update
        values : tf.Tensor
            Update values
        """
        self._check_input_values(values)
        tf.assert_equal(
            tf.shape(indices)[0],
            tf.shape(values)[0],
            "The number of indices and values should match",
        )

        self.storage.scatter_nd_update(indices, values)
