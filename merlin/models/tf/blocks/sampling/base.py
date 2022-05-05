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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Callable, Sequence

import tensorflow as tf
from merlin.models.tf.blocks.core.base import Block
from merlin.models.utils.registry import Registry, RegistryMixin


@dataclass
class Items:
    ids: tf.Tensor
    metadata: Dict[str, tf.Tensor] = field(default_factory=lambda: {})

    def embedding(self) -> tf.Tensor:
        return self.metadata[ItemSampler.ITEM_EMBEDDING_KEY]

    @property
    def has_embedding(self) -> bool:
        return ItemSampler.ITEM_EMBEDDING_KEY in self.metadata

    def with_embedding(self, embedding: tf.Tensor) -> "Items":
        self.metadata[ItemSampler.ITEM_EMBEDDING_KEY] = embedding

        return self

    def __add__(self, other):
        return Items(
            ids=_list_to_tensor([self.ids, other.ids]),
            metadata={
                key: _list_to_tensor([self.metadata[key], other.metadata[key]])
                for key, val in self.metadata.items()
            }
        )

    @property
    def shape(self) -> "Items":
        return Items(
            self.ids.shape,
            {key: val.shape for key, val in self.metadata.items()}
        )

    def __repr__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Items({self.ids}, {metadata})"

    def __str__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Items({self.ids}, {metadata})"

    def get_config(self):
        return {
            "ids": self.ids.as_list() if self.ids else None,
            "metadata": {key: val.as_list() for key, val in self.metadata.items()}
        }

    @classmethod
    def from_config(cls, config):
        ids = tf.TensorShape(config["config"]["ids"])
        metadata = {key: tf.TensorShape(val) for key, val in config["config"]["metadata"].items()}

        return cls(ids, metadata)


negative_sampling_registry: Registry = Registry.class_registry("tf.negative_sampling")


class ItemSampler(Block, RegistryMixin["ItemSampler"], abc.ABC):
    ITEM_EMBEDDING_KEY = "__item_embedding__"
    registry = negative_sampling_registry

    def __init__(
            self,
            max_num_samples: Optional[int] = None,
            **kwargs,
    ):
        super(ItemSampler, self).__init__(**kwargs)
        self.set_max_num_samples(max_num_samples)

    def call(self, items: Items, training=False) -> Items:
        if training:
            self.add(items)
        items = self.sample()

        return items

    @abc.abstractmethod
    def add(self, items: Items):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> Items:
        raise NotImplementedError()

    def _check_inputs_batch_sizes(self, items: Items):
        embeddings_batch_size = tf.shape(items.ids)[0]
        for feat_name in items.metadata:
            metadata_feat_batch_size = tf.shape(items.metadata[feat_name])[0]

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


def _list_to_tensor(input_list: List[tf.Tensor]) -> tf.Tensor:
    output: tf.Tensor

    if len(input_list) == 1:
        output = input_list[0]
    else:
        output = tf.concat(input_list, axis=0)

    return output


ItemSamplersType = Union[ItemSampler, Sequence[Union[ItemSampler, str]], str]
