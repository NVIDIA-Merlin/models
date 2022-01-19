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
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.python.ops import array_ops

from ..core import Block, ItemSampler, Sampler
from ..typing import TabularData
from ..utils.tf_utils import FIFOQueue


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class MemoryBankBlock(Block, Sampler):
    def __init__(
        self,
        num_batches: int = 1,
        key: Optional[str] = None,
        post: Optional[Block] = None,
        no_outputs: bool = False,
        stop_gradient: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key = key
        self.num_batches = num_batches
        self.queue = deque(maxlen=num_batches + 1)
        self.no_outputs = no_outputs
        self.post = post
        self.stop_gradient = stop_gradient

    def call(self, inputs: TabularData, training=True, **kwargs) -> TabularData:
        if training:
            to_add = inputs[self.key] if self.key else inputs
            self.queue.append(to_add)

        if self.no_outputs:
            return {}

        return inputs

    def sample(self) -> tf.Tensor:
        outputs = tf.concat(list(self.queue)[:-1], axis=0)

        if self.post is not None:
            outputs = self.post(outputs)

        if self.stop_gradient:
            outputs = array_ops.stop_gradient(outputs, name="memory_bank_stop_gradient")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@dataclass
class SamplingOutput:
    items_embeddings: tf.Tensor
    items_metadata: Dict[str, tf.Tensor]


class CachedBatchesSampler(ItemSampler):
    def __init__(
        self,
        batch_size: int,
        num_batches: int = 1,
        post: Optional[Block] = None,
        stop_gradient: bool = True,
        ignore_last_batch_on_sample: bool = True,
        **kwargs,
    ):
        # Reserving space for the current batch
        max_num_samples = batch_size * (num_batches + 1)
        super().__init__(max_num_samples, **kwargs)

        self.batch_size = batch_size
        self.post = post
        self.stop_gradient = stop_gradient
        self.ignore_last_batch_on_sample = ignore_last_batch_on_sample
        self.item_metadata_dtypes = None

    def _maybe_build(self, inputs):
        items_metadata = inputs["items_metadata"]
        if self.item_metadata_dtypes is None:
            self.item_metadata_dtypes = {
                feat_name: items_metadata[feat_name].dtype for feat_name in items_metadata
            }
        super()._maybe_build(inputs)

    def build(self, input_shapes):
        item_embeddings_dims = list(input_shapes["items_embeddings"][1:])
        self.item_embeddings_queue = FIFOQueue(
            capacity=self.max_num_samples,
            dims=item_embeddings_dims,
            dtype=tf.float32,
            name="item_emb",
        )

        self.items_metadata_queue = dict()
        items_metadata = input_shapes["items_metadata"]
        for feat_name in items_metadata:
            self.items_metadata_queue[feat_name] = FIFOQueue(
                capacity=self.max_num_samples,
                dims=list(items_metadata[feat_name][1:]),
                dtype=self.item_metadata_dtypes[feat_name],
                name=f"item_metadata_{feat_name}",
            )

    def call(self, inputs: TabularData, training=True):
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(
        self,
        inputs: TabularData,
        training: bool = True,
    ):
        items_embeddings = inputs["items_embeddings"]
        items_metadata = inputs["items_metadata"]

        if training:
            self.item_embeddings_queue.enqueue_many(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].enqueue_many(items_metadata[feat_name])

    def sample(self) -> TabularData:
        items_embeddings = self.item_embeddings_queue.list_all()
        items_metadata = {
            feat_name: self.items_metadata_queue[feat_name].list_all()
            for feat_name in self.items_metadata_queue
        }

        # P.s. Checks if should ignore the last batch, assuming the current batch was already added
        # to the sampler queue and that InBatchSampler(stop_gradients=False) will
        # take care of the current batch negatives
        if self.ignore_last_batch_on_sample:
            items_embeddings = items_embeddings[: -self.batch_size]
            items_metadata = {
                feat_name: items_metadata[feat_name][: -self.batch_size]
                for feat_name in items_metadata
            }

        if self.post is not None:
            items_embeddings = self.post(items_embeddings)

        if self.stop_gradient:
            items_embeddings = array_ops.stop_gradient(
                items_embeddings, name="memory_bank_stop_gradient"
            )

        return SamplingOutput(
            items_embeddings,
            items_metadata,
        )


class InBatchSampler(ItemSampler):
    def __init__(self, batch_size: int, **kwargs):
        super().__init__(max_num_samples=batch_size, **kwargs)
        self.batch_size = batch_size
        self._last_batch_items_embeddings: tf.Tensor = None
        self._last_batch_items_metadata: TabularData = None

    def call(self, inputs: TabularData, training=True):
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(self, inputs: TabularData, training=True):
        self._last_batch_items_embeddings = inputs["items_embeddings"]
        self._last_batch_items_metadata = inputs["items_metadata"]

    def sample(self) -> TabularData:
        return SamplingOutput(self._last_batch_items_embeddings, self._last_batch_items_metadata)


class UniformSampler(ItemSampler):
    def __init__(self, stop_gradient: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = stop_gradient

    def add(self, items_embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    def sample(self) -> TabularData:
        raise NotImplementedError()
