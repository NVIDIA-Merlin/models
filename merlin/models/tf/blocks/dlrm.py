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

from merlin.models.tf.blocks.core.base import Block, Debug
from merlin.models.tf.blocks.core.combinators import Filter, ParallelBlock, SequentialBlock
from merlin.models.tf.blocks.interaction import DotProductInteraction
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import EmbeddingFeatures, EmbeddingOptions
from merlin.schema import Schema, Tags


def DLRMBlock(
    schema: Schema,
    embedding_dim: int,
    bottom_block: Optional[Block] = None,
    top_block: Optional[Block] = None,
) -> SequentialBlock:
    """Builds the DLRM architecture, as proposed in the following
    `paper https://arxiv.org/pdf/1906.00091.pdf`_ [1]_.

    References
    ----------
    .. [1] Naumov, Maxim, et al. "Deep learning recommendation model for
       personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019).

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    bottom_block : Block
        The `Block` that combines the continuous features (typically a `MLPBlock`)
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

    if embedding_dim is None:
        raise ValueError("The embedding_dim is required")

    con_schema = schema.select_by_tag(Tags.CONTINUOUS).excluding_by_tag(Tags.TARGET)
    cat_schema = schema.select_by_tag(Tags.CATEGORICAL).excluding_by_tag(Tags.TARGET)

    if not len(cat_schema) > 0:
        raise ValueError("DLRM requires categorical features")

    if (
        embedding_dim is not None
        and bottom_block is not None
        and embedding_dim != bottom_block.layers[-1].units
    ):
        raise ValueError(
            f"The embedding_dim ({embedding_dim}) needs to match the "
            "last layer of bottom MLP ({bottom_block.layers[-1].units}) "
        )

    embeddings = EmbeddingFeatures.from_schema(
        cat_schema, embedding_options=EmbeddingOptions(embedding_dim_default=embedding_dim)
    )

    if len(con_schema) > 0:
        if bottom_block is None:
            raise ValueError(
                "The bottom_block is required by DLRM when "
                "continuous features are available in the schema"
            )
        con = ContinuousFeatures.from_schema(con_schema)
        bottom_block = con.connect(bottom_block)  # type: ignore
        interaction_inputs = ParallelBlock({"embeddings": embeddings, "bottom_block": bottom_block})
    else:
        interaction_inputs = embeddings  # type: ignore
        bottom_block = None

    interaction_inputs = interaction_inputs.connect(Debug())

    if not top_block:
        return interaction_inputs.connect(DotProductInteractionBlock())

    if not bottom_block:
        return interaction_inputs.connect(DotProductInteractionBlock(), top_block)

    top_block_inputs = interaction_inputs.connect_with_shortcut(
        DotProductInteractionBlock(),
        shortcut_filter=Filter("bottom_block"),
        aggregation="concat",
    )
    top_block_outputs = top_block_inputs.connect(top_block)

    return top_block_outputs


def DotProductInteractionBlock():
    return SequentialBlock(DotProductInteraction(), pre_aggregation="stack")
