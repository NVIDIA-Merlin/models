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
from typing import Optional, Union

import merlin.io
import numpy as np
import tensorflow as tf
from merlin.core.dispatch import DataFrameType
from merlin.schema import Tags
from tensorflow.python import to_dlpack

from merlin.models.tf.core import Block
from merlin.models.tf.utils.batch_utils import TFModelEncode

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
    def from_dataset(
        cls, data: merlin.io.Dataset, check_unique_ids: bool = True, **kwargs
    ) -> "IndexBlock":
        if hasattr(data, "to_ddf"):
            data = data.to_ddf()
        if check_unique_ids:
            cls._check_unique_ids(data=data)
        values = tf.convert_to_tensor(data)
        ids = tf.convert_to_tensor(data.index)

        return cls(values=values, ids=ids, **kwargs)

    @classmethod
    def from_block(
        cls, block: Block, data: merlin.io.Dataset, id_column: Optional[str] = None, **kwargs
    ) -> "IndexBlock":
        """Build candidates embeddings from applying `block` to a dataset of features `data`.

        Parameters:
        -----------
        block: Block
            The Block that returns embeddings from raw item features.
        data: merlin.io.Dataset
            Dataset containing raw item features.
        id_column: Optional[str]
            The candidates ids column name.
            Note, this will be inferred automatically if the block contains
            a schema with an item-id Tag.
        """
        if not id_column and getattr(block, "schema", None):
            tagged = block.schema.select_by_tag(Tags.ITEM_ID)
            if tagged.column_schemas:
                id_column = tagged.first.name

        model_encode = TFModelEncode(model=block, output_concat_func=np.concatenate)

        data = data.to_ddf()
        block_outputs = data.map_partitions(
            model_encode, filter_input_columns=[id_column]
        ).compute()

        block_outputs.set_index(id_column, inplace=True)

        return cls.from_dataset(block_outputs, **kwargs)

    @staticmethod
    def _check_unique_ids(data: DataFrameType):
        if data.index.nunique() != data.shape[0]:
            raise ValueError("Please make sure that `data` contains unique indices")

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

    def to_dataset(self, gpu=True) -> merlin.io.Dataset:
        if gpu:
            import cudf

            df = cudf.from_dlpack(to_dlpack(tf.convert_to_tensor(self.values)))
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(cudf.RangeIndex(0, self.values.shape[0]))
        else:
            import pandas as pd

            df = pd.DataFrame(self.values.numpy())
            df.columns = [str(col) for col in list(df.columns)]
            df.set_index(pd.RangeIndex(0, self.values.shape[0]))

        return merlin.io.Dataset(df)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TopKIndexBlock(IndexBlock):
    """Top-K index to retrieve top-k scores and indices from an item block.

    Parameters:
    -----------
        k: int
            Number of top candidates to retrieve.
        values: tf.Tensor
            The pre-computed embedddings of candidates.
        ids: tf.Tensor
            The candidates ids.
    """

    def __init__(self, k, values: tf.Tensor, ids: Optional[tf.Tensor] = None, **kwargs):
        self._k = k
        super(TopKIndexBlock, self).__init__(values, ids, **kwargs)

    @classmethod
    def from_block(
        cls,
        block: Block,
        data: merlin.io.Dataset,
        k: int = 20,
        id_column: Optional[str] = None,
        **kwargs,
    ) -> "TopKIndexBlock":
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
        k: int
            Number of top candidates to retrieve.
            Defaults to 20
         id_column: Optional[str]
            The candidates ids column name.
            Note, this will be inferred automatically if the block contains
            a schema with an item-id Tag.
        """
        return super().from_block(block=block, data=data, id_column=id_column, k=k, **kwargs)

    def call(self, inputs: tf.Tensor, k=None, **kwargs) -> Union[tf.Tensor, tf.Tensor]:
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
