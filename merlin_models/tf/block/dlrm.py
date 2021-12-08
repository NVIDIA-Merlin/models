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

from merlin_standard_lib import Schema, Tag

from ..api import merge
from ..core import Block, Filter, SequentialBlock, TabularBlock
from ..features.continuous import ContinuousFeatures
from ..features.embedding import EmbeddingFeatures, EmbeddingOptions
from ..layers import DotProductInteraction


def DLRMBlock(
    schema: Schema,
    bottom_block: Block,
    top_block: Optional[Block] = None,
    embedding_dim: Optional[int] = None,
) -> SequentialBlock:
    """Builds the DLRM archicture, as proposed in the following
    `paper https://arxiv.org/pdf/1906.00091.pdf`_ [Naumov19].

    REFERENCES:
        - [Naumov19] Naumov, Maxim, et al. "Deep learning recommendation model for
          personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019).

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    bottom_block : Block
        The `Block` that combines the continuous features (tipically a number of stacked MLP layers)
    top_block : Optional[Block], optional
        The optional `Block` that combines the outputs of bottom layer and of
        the factorization machine layer, by default None
    embedding_dim : Optional[int], optional
        Dimension of the embeddings, by default None

    Returns
    -------
    SequentialBlock
        The DLRM block

    Raises
    ------
    ValueError
        The schema is required by DLRM
    ValueError
        The bottom_block is required by DLRM
    ValueError
        The embedding_dim (X) needs to match the last layer of bottom MLP (Y).
    """
    if schema is None:
        raise ValueError("The schema is required by DLRM")
    if bottom_block is None:
        raise ValueError("The bottom_block is required by DLRM")
    if embedding_dim is not None and embedding_dim != bottom_block.layers[-1].units:
        raise ValueError(
            f"The embedding_dim ({embedding_dim}) needs to match the "
            "last layer of bottom MLP ({bottom_block.layers[-1].units})"
        )

    con_schema = schema.select_by_tag(Tag.CONTINUOUS)
    cat_schema = schema.select_by_tag(Tag.CATEGORICAL)

    top_block_inputs = {}
    if len(con_schema) > 0:
        top_block_inputs["continuous"] = ContinuousFeatures.from_schema(con_schema).connect(
            bottom_block
        )

    if len(cat_schema) > 0:
        embedding_dim = embedding_dim or bottom_block.layers[-1].units
        top_block_inputs["categorical"] = EmbeddingFeatures.from_schema(
            cat_schema, options=EmbeddingOptions(embedding_dim_default=embedding_dim)
        )

    if not top_block:
        return merge(top_block_inputs, aggregation="stack").connect(DotProductInteraction())

    dot_product = TabularBlock(aggregation="stack").connect(DotProductInteraction())
    top_block_outputs = (
        merge(top_block_inputs)
        .connect_with_shortcut(
            dot_product, shortcut_filter=Filter("continuous"), aggregation="concat"
        )
        .connect(top_block)
    )
    return top_block_outputs
