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
import abc
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.blocks.core.base import EmbeddingWithMetadata
from merlin.models.tf.typing import TabularData


class ItemSampler(abc.ABC, Layer):
    def __init__(
        self,
        max_num_samples: Optional[int] = None,
        **kwargs,
    ):
        super(ItemSampler, self).__init__(**kwargs)
        self.set_max_num_samples(max_num_samples)

    @abc.abstractmethod
    def add(self, embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> EmbeddingWithMetadata:
        raise NotImplementedError()

    def _check_inputs_batch_sizes(self, inputs: TabularData):
        embeddings_batch_size = tf.shape(inputs["embeddings"])[0]
        for feat_name in inputs["metadata"]:
            metadata_feat_batch_size = tf.shape(inputs["metadata"][feat_name])[0]

            tf.assert_equal(
                embeddings_batch_size,
                metadata_feat_batch_size,
                "The batch size (first dim) of embeddings "
                f"({int(embeddings_batch_size)}) and metadata "
                f"features ({int(metadata_feat_batch_size)}) must match.",
            )

    @property
    def required_features(self) -> List[str]:
        return []

    @property
    def max_num_samples(self) -> int:
        return self._max_num_samples

    def set_max_num_samples(self, value) -> None:
        self._max_num_samples = value
