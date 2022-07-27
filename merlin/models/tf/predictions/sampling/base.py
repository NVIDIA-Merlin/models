import abc
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import tensorflow as tf

from merlin.models.utils.registry import Registry, RegistryMixin

ITEM_EMBEDDING_KEY = "__item_embedding__"


class Items(NamedTuple):
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
            id=_list_to_tensor([self.id, other.ids]),
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


class ItemSampler(tf.keras.layers.Layer, RegistryMixin["ItemSampler"], abc.ABC):
    registry = negative_sampling_registry

    def __init__(
        self,
        max_num_samples: Optional[int] = None,
        **kwargs,
    ):
        super(ItemSampler, self).__init__(**kwargs)
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


ItemSamplersType = Union[ItemSampler, Sequence[Union[ItemSampler, str]], str]
