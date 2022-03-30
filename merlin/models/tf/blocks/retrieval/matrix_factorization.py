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
from typing import Type, Union

from merlin.models.tf.blocks.core.aggregation import ElementWiseMultiply
from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.transformations import RenameFeatures
from merlin.models.tf.blocks.retrieval.base import DualEncoderBlock
from merlin.models.tf.inputs.embedding import EmbeddingTable, EmbeddingTableOptions, InitializerFn
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin_models")


class QueryItemIdsEmbeddingsBlock(DualEncoderBlock):
    """
    This blocks embeds the query item ids into a dense representation.

    Parameters
    ----------
    schema : Schema
        The schema of the input data.
    dim: int
        The dimension of the embedding.
    query_id_tag: Union[Tags, str]
        The tag of the query item ids.
    item_id_tag: Union[Tags, str]
        The tag of the item ids.
    embedding_block_cls: Type[Block]
        The class of the embedding block.
    query_embedding_initializer: InitializerFn
        The initializer for the query embedding.
    item_embedding_initializer: InitializerFn
        The initializer for the item embedding.

    """

    def __init__(
        self,
        schema: Schema,
        dim: int,
        query_id_tag: Union[Tags, str] = Tags.USER_ID,
        item_id_tag: Union[Tags, str] = Tags.ITEM_ID,
        embedding_block_cls: Type[Block] = EmbeddingTable,
        query_embedding_initializer: InitializerFn = None,
        item_embedding_initializer: InitializerFn = None,
        **kwargs,
    ):
        embedding_options = dict(dim=dim, block_cls=embedding_block_cls)
        query_schema = schema.select_by_tag(query_id_tag)
        item_schema = schema.select_by_tag(item_id_tag)

        query_embedding = EmbeddingTableOptions(
            initializer=query_embedding_initializer, **embedding_options
        ).to_block(query_schema.first)
        item_embedding = EmbeddingTableOptions(
            initializer=item_embedding_initializer, **embedding_options
        ).to_block(item_schema.first)

        rename_features = RenameFeatures(
            {query_id_tag: "query", item_id_tag: "item"}, schema=schema
        )
        post = kwargs.pop("post", None)
        if post:
            post = rename_features.connect(post)
        else:
            post = rename_features

        super().__init__(query_embedding, item_embedding, post=post, **kwargs)

    def export_embedding_table(self, table_name: Union[str, Tags], export_path: str, gpu=True):
        if table_name in ("item", Tags.ITEM_ID):
            return self.item_block().block.export_embedding_table(table_name, export_path, gpu=gpu)

        return self.query_block().block.export_embedding_table(table_name, export_path, gpu=gpu)

    def embedding_table_df(self, table_name: Union[str, Tags], gpu=True):
        if table_name in ("item", Tags.ITEM_ID):
            return self.item_block().block.embedding_table_df(table_name, gpu=gpu)

        return self.query_block().block.embedding_table_df(table_name, gpu=gpu)


def MatrixFactorizationBlock(
    schema: Schema,
    dim: int,
    query_id_tag: Union[Tags, str] = Tags.USER_ID,
    item_id_tag: Union[Tags, str] = Tags.ITEM_ID,
    query_embedding_initializer: InitializerFn = None,
    item_embedding_initializer: InitializerFn = None,
    aggregation=ElementWiseMultiply(),
    **kwargs,
) -> QueryItemIdsEmbeddingsBlock:
    """Block that performs matrix factorization.

    Parameters
    ----------
    schema : Schema
        The schema of the input data.
    dim: int
        The dimension of the embedding.
    query_id_tag: Union[Tags, str]
        The tag of the query item ids.
    item_id_tag: Union[Tags, str]
        The tag of the item ids.
    query_embedding_initializer
        The initializer for the query embedding.
    item_embedding_initializer
        The initializer for the item embedding.
    aggregation: ElementWiseMultiply
        The aggregation function to use.

    Returns
    -------
    QueryItemIdsEmbeddingsBlock
    """

    return QueryItemIdsEmbeddingsBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        query_embedding_initializer=query_embedding_initializer,
        item_embedding_initializer=item_embedding_initializer,
        aggregation=aggregation,
        **kwargs,
    )
