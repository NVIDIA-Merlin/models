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

EMBEDDING_KEY = "__embedding__"


class Candidate(NamedTuple):
    """Store candidate id and their metadata

    Parameters
    ----------
    id : tf.Tensor
        The tensor of item ids
    sampling_prob : tf.Tensor
        Useful for logQ correction, based on the sampling distribution
    metadata:
        dictionary of tensors containing meta information
        about items such as item embeddings and item category
    """

    id: tf.Tensor
    metadata: Dict[str, tf.Tensor]
    sampling_prob: Optional[tf.Tensor] = None

    @property
    def embedding(self) -> tf.Tensor:
        return self.metadata[EMBEDDING_KEY]

    @property
    def has_embedding(self) -> bool:
        return EMBEDDING_KEY in self.metadata

    def with_embedding(self, embedding: tf.Tensor) -> "Candidate":
        self.metadata[EMBEDDING_KEY] = embedding

        return self

    def with_sampling_prob(self, sampling_prob: tf.Tensor) -> "Candidate":
        return Candidate(id=self.id, metadata=self.metadata, sampling_prob=sampling_prob)

    def __add__(self, other):
        metadata = {}
        for key in self.metadata:
            if key in other.metadata:
                metadata[key] = _list_to_tensor([self.metadata[key], other.metadata[key]])

        return Candidate(id=_list_to_tensor([self.id, other.id]), metadata=metadata)

    @property
    def shape(self) -> "Candidate":
        return Candidate(
            id=self.id.shape, metadata={key: val.shape for key, val in self.metadata.items()}
        )

    def __repr__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Candidate({self.id}, {self.sampling_prob}, {metadata})"

    def __str__(self):
        metadata = {key: str(val) for key, val in self.metadata.items()}

        return f"Candidate({self.id}, {self.sampling_prob}, {metadata})"

    def __eq__(self, other) -> bool:
        if self.id.shape != other.id.shape:
            return False

        return self.id.ref() == other.id.ref()

    def get_config(self):
        return {
            "id": self.id,
            "sampling_prob": self.sampling_prob,
            "metadata": self.metadata,
        }

    @classmethod
    def from_config(cls, config):
        ids = config["config"]["id"]
        sampling_prob = config["config"]["sampling_prob"]
        metadata = config["config"]["metadata"]

        return cls(ids, sampling_prob, metadata)


negative_sampling_registry: Registry = Registry.class_registry("tf.negative_sampling")


class CandidateSampler(tf.keras.layers.Layer, RegistryMixin["CandidateSampler"], abc.ABC):
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
        super(CandidateSampler, self).__init__(**kwargs)
        self.set_max_num_samples(max_num_samples)

    def call(
        self, items: Candidate, features=None, targets=None, training=False, testing=False
    ) -> Candidate:
        if training:
            self.add(items)
        items = self.sample()

        return items

    @abc.abstractmethod
    def add(self, items: Candidate):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> Candidate:
        raise NotImplementedError()

    def with_sampling_probs(self, items: Candidate) -> Candidate:
        return items

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


ItemSamplersType = Union[CandidateSampler, Sequence[Union[CandidateSampler, str]], str]


def parse_negative_samplers(negative_sampling: ItemSamplersType):
    """
    Parse the negative sampling strategies and returns
    the corresponding list of samplers.
    """
    if negative_sampling is None:
        negative_sampling = []
    if not isinstance(negative_sampling, (list, tuple)):
        negative_sampling = [negative_sampling]
    negative_sampling = [CandidateSampler.parse(s) for s in list(negative_sampling)]

    return negative_sampling
