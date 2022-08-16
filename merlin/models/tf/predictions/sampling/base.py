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
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import tensorflow as tf

from merlin.models.utils.registry import Registry, RegistryMixin

ITEM_EMBEDDING_KEY = "__item_embedding__"


class Items(NamedTuple):
    """Storea item ids and their metadata

    Parameters
    ----------
    id : tf.Tensor
        The tensor of item ids
    metadata:
        dictionary of tensors containing meta information
        about items such as item embeddings and item category
    """

    id: tf.Tensor
    metadata: Dict[str, tf.Tensor]

    def embedding(self) -> tf.Tensor:
        return self.metadata[ITEM_EMBEDDING_KEY]

    @property
    def has_embedding(self) -> bool:
        return ITEM_EMBEDDING_KEY in self.metadata

    def with_embedding(self, embedding: tf.Tensor) -> "Items":
        self.metadata[ITEM_EMBEDDING_KEY] = embedding

        return self

    def __add__(self, other):
        return Items(
            id=_list_to_tensor([self.id, other.id]),
            metadata={
                key: _list_to_tensor([self.metadata[key], other.metadata[key]])
                for key, val in self.metadata.items()
            },
        )

    @property
    def shape(self) -> "Items":
        return Items(self.id.shape, {key: val.shape for key, val in self.metadata.items()})

    def __repr__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Items({self.id}, {metadata})"

    def __str__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Items({self.id}, {metadata})"

    def get_config(self):
        return {
            "ids": self.id.as_list() if self.id else None,
            "metadata": {key: val.as_list() for key, val in self.metadata.items()},
        }

    @classmethod
    def from_config(cls, config):
        ids = tf.TensorShape(config["config"]["id"])
        metadata = {key: tf.TensorShape(val) for key, val in config["config"]["metadata"].items()}

        return cls(ids, metadata)


negative_sampling_registry: Registry = Registry.class_registry("tf.negative_sampling")


class ItemSamplerV2(tf.keras.layers.Layer, RegistryMixin["ItemSampler"], abc.ABC):
    """Base-class for negative sampling

    Parameters
    ----------
    max_num_samples : int
        The number of maximum samples to store

    Returns
    -------
    Items
        The sampled ids and their metadata
    """

    registry = negative_sampling_registry

    def __init__(
        self,
        max_num_samples: Optional[int] = None,
        **kwargs,
    ):
        super(ItemSamplerV2, self).__init__(**kwargs)
        self.set_max_num_samples(max_num_samples)

    def call(
        self, items: Items, features=None, targets=None, training=False, testing=False
    ) -> Items:
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


ItemSamplersType = Union[ItemSamplerV2, Sequence[Union[ItemSamplerV2, str]], str]


def parse_negative_samplers(negative_sampling: ItemSamplersType):
    """
    Parse the negative sampling strategies and returns
    the corresponding list of samplers.
    """
    if not isinstance(negative_sampling, (list, tuple)):
        negative_sampling = [negative_sampling]
    negative_sampling = [ItemSamplerV2.parse(s) for s in list(negative_sampling)]
    return negative_sampling
