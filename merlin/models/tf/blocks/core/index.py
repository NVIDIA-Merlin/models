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

import numpy as np
import tensorflow as tf
from tensorflow.python import to_dlpack

import merlin.io
from merlin.core.dispatch import DataFrameType
from merlin.models.tf.blocks.core.base import Block, PredictionOutput
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.batch_utils import TFModelEncode
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class IndexBlock(Block):
    def __init__(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None, **kwargs):
        super(IndexBlock, self).__init__(**kwargs)
        self.values = tf.Variable(
            values,
            name="values",
            trainable=False,
            dtype=tf.float32,
            validate_shape=False,
            shape=tf.TensorShape([None, tf.shape(values)[-1]]),
        )
        if ids is not None:
            id_dtype = ids.dtype
        else:
            id_dtype = tf.int64

        self.ids = tf.Variable(
            ids,
            name="ids",
            trainable=False,
            dtype=id_dtype,
            validate_shape=False,
            shape=tf.TensorShape([None]),
        )

    @classmethod
    def from_dataset(
        cls, data: merlin.io.Dataset, check_unique_ids: bool = True, **kwargs
    ) -> "IndexBlock":
        ids, values = cls.extract_ids_embeddings(data, check_unique_ids)
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
        embedding_df = cls.get_candidates_dataset(block, data, id_column)
        return cls.from_dataset(embedding_df, **kwargs)

    @staticmethod
    def _check_unique_ids(data: DataFrameType):
        if data.index.to_series().nunique() != data.shape[0]:
            raise ValueError("Please make sure that `data` contains unique indices")

    @classmethod
    def extract_ids_embeddings(cls, data: merlin.io.Dataset, check_unique_ids: bool = True):
        if hasattr(data, "to_ddf"):
            data = data.to_ddf()
        if check_unique_ids:
            cls._check_unique_ids(data=data)
        values = tf_utils.df_to_tensor(data)
        ids = tf_utils.df_to_tensor(data.index)

        if len(ids.shape) == 2:
            ids = tf.squeeze(ids)

        return ids, values

    @classmethod
    def get_candidates_dataset(
        cls, block: Block, data: merlin.io.Dataset, id_column: Optional[str] = None
    ):
        if not id_column and getattr(block, "schema", None):
            tagged = block.schema.select_by_tag(Tags.ITEM_ID)
            if tagged.column_schemas:
                id_column = tagged.first.name

        model_encode = TFModelEncode(model=block, output_concat_func=np.concatenate)

        data = data.to_ddf()
        embedding_ddf = data.map_partitions(model_encode, filter_input_columns=[id_column])
        embedding_df = embedding_ddf.compute(scheduler="synchronous")

        embedding_df.set_index(id_column, inplace=True)
        return embedding_df

    def update_from_block(
        self,
        block: Block,
        data: merlin.io.Dataset,
        id_column: Optional[str] = None,
        check_unique_ids: bool = True,
    ):
        embedding_df = IndexBlock.get_candidates_dataset(block, data, id_column)
        ids, embeddings = IndexBlock.extract_ids_embeddings(embedding_df, check_unique_ids)
        self.update(embeddings, ids)

    def update(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None):
        if len(tf.shape(values)) != 2:
            raise ValueError(f"The candidates embeddings tensor must be 2D (got {values.shape}).")
        _ids: tf.Tensor = ids if ids is not None else tf.range(values.shape[0])
        self.ids.assign(_ids)
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
        self.false_negatives_score = MIN_FLOAT

    @classmethod
    def from_block(  # type: ignore
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
            The Block that returns embeddings from raw item features.
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

    def call_outputs(
        self, outputs: PredictionOutput, training=False, **kwargs
    ) -> "PredictionOutput":
        """
        Retrieve top-k negative scores for evaluation.

        Parameters
        ----------
        predictions: tf.Tensor
            Tensor of pre-computed positive scores.
            If`training=True`, the first column of predictions is expected
            to be positive scores and the remaining sampled negatives are ignored.

        Returns
        -------
        targets, predictions: tf.Tensor, tf.Tensor
            2D Tensors with the one-hot representation of true targets and
            the scores for the top-k implicit negatives.
        """
        queries = self.context["query"]
        pred_top_scores, top_ids = self(queries, k=self._k)

        targets_sorted = tf.cast(
            tf.expand_dims(outputs.positive_item_ids, -1) == top_ids, tf.float32
        )
        targets_sorted = tf.reshape(targets_sorted, tf.shape(pred_top_scores))

        label_relevant_counts = tf.ones([tf.shape(targets_sorted)[0]])

        return outputs.copy_with_updates(
            predictions=pred_top_scores,
            targets=targets_sorted,
            label_relevant_counts=label_relevant_counts,
        )

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return tf.TensorShape((batch_size, self._k)), tf.TensorShape((batch_size, self._k))
