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

from merlin.models.tf.blocks.interaction import DotProductInteraction
from merlin.models.tf.core.aggregation import StackFeatures
from merlin.models.tf.core.base import Block, Debug
from merlin.models.tf.core.combinators import Filter, ParallelBlock, SequentialBlock
from merlin.models.tf.core.transformations import AsRaggedFeatures
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import EmbeddingOptions, Embeddings
from merlin.schema import Schema, Tags


def DLRMBlock(
    schema: Schema,
    embedding_dim: int,
    embedding_options: EmbeddingOptions = None,
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
    embedding_options : EmbeddingOptions
        Options for the input embeddings.
        - embedding_dim_default: int - Default dimension of the embedding
        table, when the feature is not found in ``embedding_dims``, by default 64
        - infer_embedding_sizes : bool, Automatically defines the embedding
        dimension from the feature cardinality in the schema, by default False,
        which needs to be kept False for the DLRM architecture.

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

    if embedding_options is not None:
        embedding_options.embedding_dim_default = embedding_dim
        embedding_options.infer_embedding_sizes = False
    else:
        embedding_options = EmbeddingOptions(embedding_dim_default=embedding_dim)

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

    embeddings_kwargs = dict(
        sequence_combiner=embedding_options.combiner,
        embedding_dims=embedding_options.embedding_dims,
        embedding_dim_default=embedding_options.embedding_dim_default,
        infer_embedding_sizes=embedding_options.infer_embedding_sizes,
        infer_embedding_sizes_multiplier=embedding_options.infer_embedding_sizes_multiplier,
    )
    embeddings_kwargs["infer_embeddings_ensure_dim_multiple_of_8"] = (
        embedding_options.infer_embeddings_ensure_dim_multiple_of_8,
    )
    embeddings = SequentialBlock(
        AsRaggedFeatures(),
        Embeddings(
            cat_schema,
            **embeddings_kwargs,
        ),
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
    return SequentialBlock(StackFeatures(axis=1), DotProductInteraction())
