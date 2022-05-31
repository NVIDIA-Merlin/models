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
import logging
from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf

from merlin.models.tf.blocks.core.aggregation import CosineSimilarity
from merlin.models.tf.blocks.core.transformations import RenameFeatures
from merlin.models.tf.blocks.retrieval.base import DualEncoderBlock
from merlin.models.tf.inputs.embedding import EmbeddingFeatures, EmbeddingOptions
from merlin.schema import Schema, Tags, TagsType

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class QueryItemIdsEmbeddingsBlock(DualEncoderBlock):
    """
    An encoder for user ids and item ids

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    dim : int
        Dimension of the user and item embeddings
    query_id_tag : TagsType, optional
        The tag to select the user id feature, by default `Tags.USER_ID`
    item_id_tag : TagsType, optional
        The tag to select the item id feature, by default `Tags.ITEM_ID`
    embeddings_initializers : Optional[
            Union[Dict[str, Callable[[Any], None]], Callable[[Any], None]]
        ],
        An initializer function or a dict where keys are feature names and values are
        callable to initialize embedding tables
    embeddings_l2_reg: float = 0.0
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
    """

    def __init__(
        self,
        schema: Schema,
        dim: int,
        query_id_tag: TagsType = Tags.USER_ID,
        item_id_tag: TagsType = Tags.ITEM_ID,
        embeddings_initializers: Optional[
            Union[Dict[str, Callable[[Any], None]], Callable[[Any], None]]
        ] = None,
        embeddings_l2_reg: float = 0.0,
        **kwargs,
    ):
        query_schema = schema.select_by_tag(query_id_tag)
        item_schema = schema.select_by_tag(item_id_tag)
        embedding_options = EmbeddingOptions(
            embedding_dim_default=dim,
            embeddings_l2_reg=embeddings_l2_reg,
        )
        if embeddings_initializers:
            embedding_options.embeddings_initializers = embeddings_initializers

        rename_features = RenameFeatures(
            {query_id_tag: "query", item_id_tag: "item"}, schema=schema
        )
        post = kwargs.pop("post", None)
        if post:
            post = rename_features.connect(post)
        else:
            post = rename_features

        embedding_features = lambda s: EmbeddingFeatures.from_schema(  # noqa
            s, embedding_options=embedding_options, aggregation="concat"
        )

        self.embeddings = dict()
        self.embeddings[str(Tags.USER_ID)] = embedding_features(query_schema)
        self.embeddings[str(Tags.ITEM_ID)] = embedding_features(item_schema)

        super().__init__(  # type: ignore
            self.embeddings[str(Tags.USER_ID)],
            self.embeddings[str(Tags.ITEM_ID)],
            post=post,
            **kwargs,
        )

    def export_embedding_table(
        self, table_name: Union[str, Tags], export_path: str, l2_normalization=False, gpu=True
    ):
        return self.embeddings[str(table_name)].export_embedding_table(
            table_name, export_path, l2_normalization=l2_normalization, gpu=gpu
        )

    def embedding_table_df(self, table_name: Union[str, Tags], l2_normalization=False, gpu=True):
        return self.embeddings[str(table_name)].embedding_table_df(
            table_name, l2_normalization=l2_normalization, gpu=gpu
        )

    def get_embedding_table(self, table_name: Union[str, Tags], l2_normalization=False):
        return self.embeddings[str(table_name)].get_embedding_table(
            table_name, l2_normalization=l2_normalization
        )


def MatrixFactorizationBlock(
    schema: Schema,
    dim: int,
    query_id_tag=Tags.USER_ID,
    item_id_tag=Tags.ITEM_ID,
    embeddings_initializers: Optional[
        Union[Dict[str, Callable[[Any], None]], Callable[[Any], None]]
    ] = None,
    embeddings_l2_reg: float = 0.0,
    aggregation=CosineSimilarity(),
    **kwargs,
):
    """
    Returns a block for Matrix Factorization, which created the user and
    item embeddings based on the `schema` and computes the dot product
    between user and item L2-norm embeddings

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    dim : int
        Dimension of the user and item embeddings
    query_id_tag : _type_, optional
        The tag to select the user id feature, by default `Tags.USER_ID`
    item_id_tag : _type_, optional
        The tag to select the item id feature, by default `Tags.ITEM_ID`
    embeddings_initializers : Optional[Dict[str, Callable[[Any], None]]] = None
        An initializer function or a dict where keys are feature names and values are
        callable to initialize embedding tables
    embeddings_l2_reg: float = 0.0
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
    aggregation : _type_, optional
        Aggregation of the user and item embeddings, by default CosineSimilarity()

    Returns
    -------
    QueryItemIdsEmbeddingsBlock
        A block that encodes user ids and item ids into embeddings and computes their
        dot product
    """
    return QueryItemIdsEmbeddingsBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        embeddings_initializers=embeddings_initializers,
        embeddings_l2_reg=embeddings_l2_reg,
        aggregation=aggregation,
        **kwargs,
    )
