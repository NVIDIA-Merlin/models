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
from collections import defaultdict, deque
from typing import Optional

import tensorflow as tf
from tensorflow.python.ops import array_ops

from ..core import Block, ItemSampler, Sampler
from ..typing import TabularData


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


class CachedBatchesSampler(ItemSampler):
    def __init__(
        self,
        # batch_size: int,
        num_batches: int = 1,
        key: Optional[str] = None,
        post: Optional[Block] = None,
        stop_gradient: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key = key
        # Adds one batch to account for the current batch
        # (assuming calls in this order each batch sampler.add(), sampler.sample())
        self._num_batches = num_batches + 1
        self.item_embeddings_queue = deque(maxlen=self._num_batches)
        self.items_metadata_queue = defaultdict(lambda: deque(maxlen=self._num_batches))
        # self.num_negatives = batch_size * num_batches
        self.post = post
        self.stop_gradient = stop_gradient

    def add(
        self,
        items_embeddings: tf.Tensor,
        items_metadata: TabularData,
        training=True,
    ):
        if training:
            self.item_embeddings_queue.append(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].append(items_metadata[feat_name])

    def sample(self) -> TabularData:
        # P.s. Always ignores the last batch, assuming the current batch was already added
        # to the sampler queue and that InBatchSampler(stop_gradients=False) will
        # take care of the current batch negatives
        if len(list(self.item_embeddings_queue)[:-1]) == 0:
            return {}
        elif len(list(self.item_embeddings_queue)[:-1]) == 1:
            items_embeddings = list(self.item_embeddings_queue)[0]
            items_metadata = {
                feat_name: list(self.items_metadata_queue[feat_name])[0]
                for feat_name in self.items_metadata_queue
            }
        else:
            items_embeddings = tf.concat(list(self.item_embeddings_queue)[:-1], axis=0)
            # Concatenating item metadata features from all cached batches
            items_metadata = {}
            for feat_name in self.items_metadata_queue:
                items_metadata[feat_name] = tf.concat(
                    [
                        batch_feat_metadata
                        for batch_feat_metadata in list(self.items_metadata_queue[feat_name])[:-1]
                    ],
                    axis=0,
                )

        if self.post is not None:
            items_embeddings = self.post(items_embeddings)

        if self.stop_gradient:
            # ERROR: This stop_gradient() is causing an error on embeddings cached from past batches
            items_embeddings = array_ops.stop_gradient(
                items_embeddings, name="memory_bank_stop_gradient"
            )

        return {"items_embeddings": items_embeddings, "items_metadata": items_metadata}


class InBatchSampler(ItemSampler):
    def __init__(self, stop_gradient: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = stop_gradient
        self._last_batch_items_embeddings: tf.Tensor = None
        self._last_batch_items_metadata: TabularData = None

    def add(self, items_embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        if training:
            self._last_batch_items_embeddings = items_embeddings
            self._last_batch_items_metadata = items_metadata

    def sample(self) -> TabularData:
        last_batch_items_embeddings = self._last_batch_items_embeddings
        if self.stop_gradient:
            last_batch_items_embeddings = array_ops.stop_gradient(
                last_batch_items_embeddings, name="in_batch_sampler_stop_gradient"
            )

        return {
            "items_embeddings": last_batch_items_embeddings,
            "items_metadata": self._last_batch_items_metadata,
        }


class UniformSampler(ItemSampler):
    def __init__(self, stop_gradient: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = stop_gradient

    def add(self, items_embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    def sample(self) -> TabularData:
        raise NotImplementedError()
