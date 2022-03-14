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
from typing import Any, Callable, Dict, Optional

from merlin.models.tf.blocks.core.aggregation import ElementWiseMultiply
from merlin.models.tf.blocks.core.transformations import RenameFeatures
from merlin.models.tf.features.embedding import EmbeddingFeatures, EmbeddingOptions
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin_models")


def QueryItemIdsEmbeddingsBlock(
    schema: Schema,
    dim: int,
    query_id_tag=Tags.USER_ID,
    item_id_tag=Tags.ITEM_ID,
    embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
    **kwargs,
):
    """Buidls Query and Item ids embeddings

    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features
    dim: int
        The dimension of the embeddings.
    query_id_tag : Tag
        The tag to select query features, by default `Tags.USER`
    item_id_tag : Tag
        The tag to select item features, by default `Tags.ITEM`
    embeddings_initializers: Dict[str, Callable[[Any], None]]
        A dictionary of initializers for embeddings.
    """
    query_item_schema = schema.select_by_tag(query_id_tag) + schema.select_by_tag(item_id_tag)
    embedding_options = EmbeddingOptions(
        embedding_dim_default=dim, embeddings_initializers=embeddings_initializers
    )

    rename_features = RenameFeatures({query_id_tag: "query", item_id_tag: "item"}, schema=schema)
    post = kwargs.pop("post", None)
    if post:
        post = rename_features.connect(post)
    else:
        post = rename_features

    embeddings_blocks = EmbeddingFeatures.from_schema(
        query_item_schema, post=post, embedding_options=embedding_options, **kwargs
    )

    return embeddings_blocks


def MatrixFactorizationBlock(
    schema: Schema,
    dim: int,
    query_id_tag=Tags.USER_ID,
    item_id_tag=Tags.ITEM_ID,
    embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
    aggregation=ElementWiseMultiply(),
    **kwargs,
):
    return QueryItemIdsEmbeddingsBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        embeddings_initializers=embeddings_initializers,
        aggregation=aggregation,
        **kwargs,
    )
