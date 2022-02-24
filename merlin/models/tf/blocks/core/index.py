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

import merlin.io
import numpy as np
import tensorflow as tf

from merlin.models.tf.core import Block
from merlin.models.tf.utils.batch import TFModelEncode

"""
This would be useful for instance to convert the item-tower.
We could integrate this into the Block-class.

two_tower_block = ...
topk_index = TopKIndex.from_block(two_tower_block.item_block(), item_dataset)

recommender = two_tower_block.query_block().connect(topk_index)



"""


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class IndexBlock(Block):
    def __init__(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None, **kwargs):
        super(IndexBlock, self).__init__(**kwargs)
        self.values = values
        self.ids = ids

    @classmethod
    def from_dataset(cls, data: merlin.io.Dataset, id_column: str, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_block(cls, block: Block, data: merlin.io.Dataset, id_column: str, **kwargs):
        raise NotImplementedError()

    def update(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None):
        if len(tf.shape(values)) != 2:
            raise ValueError(f"The candidates embeddings tensor must be 2D (got {values.shape}).")
        if not ids:
            ids = tf.range(values.shape[0])

        self.ids.assign(ids)
        self.values.assign(values)
        return self

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.values[inputs]

    def to_dataset(self, **kwargs) -> merlin.io.Dataset:
        raise NotImplementedError()


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TopKIndex(IndexBlock):
    """
    Top-K index to retrieve top-k scores and indices from an item block.

    Parameters:
    -----------
        k:
        values: tf.Tensor
            The pre-computed embedddings of candidates.
        ids: tf.Tensor
            The candidates ids.
    """

    def __init__(self, k, values: tf.Tensor, ids: Optional[tf.Tensor] = None, **kwargs):
        self._k = k
        super(TopKIndex, self).__init__(values, ids, **kwargs)

    @classmethod
    def from_block(
        cls,
        block: Block,
        output_dim: int,
        data: merlin.io.Dataset,
        id_column: str,
        k: int = 20,
        **kwargs,
    ):
        """
        class method to build candidates embeddings from
        applying `block` to a dataset of features `data`

        Parameters:
        -----------
        block: Block
            The Block that retruns embeddings from raw item features.
        output_dim: int
            The output dimension of `block`.
        data: merlin.io.Dataset
            Dataset containing raw item features.
        id_column: str
            The candidates ids column name.
        k: int
            Number of top candidates to retrieve.
            Defaults to 20
        """
        output_names = [str(i) for i in range(output_dim)]
        model_encode = TFModelEncode(
            model=block, output_names=output_names, output_concat_func=np.concatenate
        )

        data = data.to_ddf()
        ids = data[id_column].compute()
        values = data.map_partitions(model_encode).compute()

        values = tf.convert_to_tensor(values[output_names])
        ids = tf.convert_to_tensor(ids)
        return cls(k, values, ids, **kwargs)

    def call(self, inputs: tf.Tensor, k=None, **kwargs) -> tf.Tensor:
        """
        Compute Top-k scores and related indices from query inputs

        Parameters:
        ----------
        inputs: tf.Tensor
            Tensor of pre-computed query embeddings.
        k: int
            Number of top candidates to retrieve
            Defaults to constructor `_k` parameter.
        Returns
        -------
        top_scores, top_indices: tf.Tensor, tf.Tensor
            2D Tensors with the scores for the top-k candidates and related ids.
        """
        k = k if k is not None else self._k
        scores = tf.matmul(inputs, self.values, transpose_b=True)
        top_scores, top_indices = tf.math.top_k(scores, k=k)
        top_indices = tf.gather(self.ids, top_indices)
        return top_scores, top_indices
