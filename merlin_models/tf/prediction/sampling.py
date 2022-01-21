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

from typing import Optional

import tensorflow as tf

from ..core import Block, ItemSampler, ItemSamplerData
from ..typing import TabularData
from ..utils.tf_utils import FIFOQueue


class InBatchSampler(ItemSampler):
    def __init__(self, batch_size: int, **kwargs):
        super().__init__(max_num_samples=batch_size, **kwargs)
        self.batch_size = batch_size
        self._last_batch_items_embeddings: tf.Tensor = None
        self._last_batch_items_metadata: TabularData = None

    def call(self, inputs: TabularData, training=True) -> ItemSamplerData:
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(self, inputs: TabularData, training=True) -> None:
        self._check_inputs_batch_sizes(inputs)
        self._last_batch_items_embeddings = inputs["items_embeddings"]
        self._last_batch_items_metadata = inputs["items_metadata"]

    def sample(self) -> ItemSamplerData:
        return ItemSamplerData(self._last_batch_items_embeddings, self._last_batch_items_metadata)


class CachedBatchesSampler(ItemSampler):
    def __init__(
        self,
        batch_size: int,
        num_batches_to_cache: int = 1,
        post: Optional[Block] = None,
        ignore_last_batch_on_sample: bool = True,
        **kwargs,
    ):
        assert batch_size > 0
        assert num_batches_to_cache > 0

        max_num_samples = batch_size * num_batches_to_cache
        super().__init__(max_num_samples, **kwargs)

        self.batch_size = batch_size
        self.post = post
        self.ignore_last_batch_on_sample = ignore_last_batch_on_sample
        self.item_metadata_dtypes = None

        self._last_batch_size = 0
        self._item_embeddings_queue = None

    def _maybe_build(self, inputs: TabularData) -> None:
        items_metadata = inputs["items_metadata"]
        if self.item_metadata_dtypes is None:
            self.item_metadata_dtypes = {
                feat_name: items_metadata[feat_name].dtype for feat_name in items_metadata
            }
        super()._maybe_build(inputs)

    def build(self, input_shapes: TabularData) -> None:
        # Reserving additional space for the current batch when ignore_last_batch_on_sample=True
        queue_size = self.max_num_samples
        if self.ignore_last_batch_on_sample:
            queue_size += self.batch_size

        item_embeddings_dims = list(input_shapes["items_embeddings"][1:])
        self._item_embeddings_queue = FIFOQueue(
            capacity=queue_size,
            dims=item_embeddings_dims,
            dtype=tf.float32,
            name="item_emb",
        )

        self.items_metadata_queue = dict()
        items_metadata = input_shapes["items_metadata"]
        for feat_name in items_metadata:
            self.items_metadata_queue[feat_name] = FIFOQueue(
                capacity=queue_size,
                dims=list(items_metadata[feat_name][1:]),
                dtype=self.item_metadata_dtypes[feat_name],
                name=f"item_metadata_{feat_name}",
            )

    def _check_built(self) -> None:
        if self._item_embeddings_queue is None:
            raise Exception(
                "The CachedBatchesSampler layer was not built yet. "
                "You need to call() that layer at least once before "
                "so that it is built before calling add() or sample() directly"
            )

    def call(self, inputs: TabularData, training=True) -> ItemSamplerData:
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(
        self,
        inputs: TabularData,
        training: bool = True,
    ) -> None:
        self._check_built()
        self._check_inputs_batch_sizes(inputs)

        items_embeddings = inputs["items_embeddings"]
        items_metadata = inputs["items_metadata"]
        if training:
            self._item_embeddings_queue.enqueue_many(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].enqueue_many(items_metadata[feat_name])

            self._last_batch_size = tf.shape(items_embeddings)[0]

    def sample(self) -> ItemSamplerData:
        self._check_built()
        items_embeddings = self._item_embeddings_queue.list_all()
        items_metadata = {
            feat_name: self.items_metadata_queue[feat_name].list_all()
            for feat_name in self.items_metadata_queue
        }

        # Checks if should ignore the last batch, assuming the current batch was already added
        # to the sampler queue and that InBatchSampler(stop_gradients=False) will
        # take care of the current batch negatives
        if self.ignore_last_batch_on_sample:
            items_embeddings = items_embeddings[: -self._last_batch_size]
            items_metadata = {
                feat_name: items_metadata[feat_name][: -self._last_batch_size]
                for feat_name in items_metadata
            }

        if self.post is not None:
            items_embeddings = self.post(items_embeddings)

        return ItemSamplerData(
            items_embeddings,
            items_metadata,
        )


class UniformSampler(ItemSampler):
    def __init__(self, stop_gradient: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = stop_gradient

    def add(self, items_embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    def sample(self) -> TabularData:
        raise NotImplementedError()
