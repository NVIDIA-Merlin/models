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
from typing import List, Optional, Union

from merlin.models.tf.blocks.core.aggregation import ElementWiseMultiply
from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.combinators import ParallelBlock
from merlin.models.tf.blocks.retrieval.matrix_factorization import (
    MatrixFactorizationBlock,
    QueryItemIdsEmbeddingsBlock,
)
from merlin.models.tf.blocks.mlp import MLPBlock
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.inputs.embedding import EmbeddingOptions
from merlin.models.tf.models.base import Model
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.schema import Schema
import warnings


def NCFModel(
    schema: Schema,
    embedding_dim: int,
    mlp_block: Block,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    embeddings_l2_reg: float = 0.0,
    **kwargs
) -> Model:
    """NCF-model architecture.

    Example Usage::
        ncf = NCFModel(schema, embedding_dim=64, mlp_block=MLPBlock([256, 64]))
        ncf.compile(optimizer="adam")
        ncf.fit(train_data, epochs=10)

    References
    ----------
    [1] Xiangnan,
        He, et al. "Neural Collaborative Filtering."
        arXiv:1708.05031 (2017).
    [2] Steffen, Rendle et al.
        "Neural Collaborative Filtering vs. Matrix Factorization Revisited."
        arXiv:2005.09683(2020)

    Notes
    -----
    We note that  Rendle et al. [2] showed that properly initialized MF
    significantly outperforms the NCF model.

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    embedding_dim : int
        Dimension of the embeddings
    mlp_block : MLPBlock
        Stack of MLP layers to learn non-linear interactions from data.
    prediction_tasks: optional
        The prediction tasks to be used, by default this will be inferred from the Schema.
    embeddings_l2_reg: float = 0.0
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
    Returns
    -------
    Model

    """

    mlp_branch = QueryItemIdsEmbeddingsBlock(
        schema, dim=embedding_dim, embeddings_l2_reg=embeddings_l2_reg
    ).connect(mlp_block)
    mf_branch = MatrixFactorizationBlock(
        schema,
        dim=embedding_dim,
        aggregation=ElementWiseMultiply(),
        embeddings_l2_reg=embeddings_l2_reg,
        **kwargs,
    )

    ncf = ParallelBlock({"mf": mf_branch, "mlp": mlp_branch}, aggregation="concat")

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
    model = Model(ncf, prediction_tasks)

    return model

def WideAndDeepModel(
    schema: Schema,
    embedding_dim: int,
    wide_schema: Optional[Schema] = None,
    deep_schema: Optional[Schema] = None,
    deep_block: Optional[Block] = None,
    deep_input_block: Optional[Block] = None,
    wide_input_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    embedding_option_kwargs: dict = {},
    **kwargs
 ) -> Model:
    """Wide-and-Deep-model architecture.

    Example Usage::
        wide_deep = WideAndDeepModel(schema, embedding_dim=32)
        wide_deep.compile(optimizer="adam")
        wide_deep.fit(train_data, epochs=10)

    References
    ----------
    [1] Cheng, Koc, Harmsen, Shaked, Chandra, Aradhye, Anderson et al. "Wide & deep learning for 
    recommender systems." In Proceedings of the 1st workshop on deep learning for recommender 
    systems, pp. 7-10. (2016).
    
    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    embedding_dim : int
        Dimension of the embeddings
    wide_schema : optional
        The 'Schema' of input features for wide model, by default all features would be sent
        to wide model, if specified, only features in wide_schema would be sent to wide model
    deep_schema : optional
        The 'Schema' of input features for deep model, by default all features would be sent to
        deep model. deep_schema and wide_schema could contain the same features
    prediction_tasks: optional
        The prediction tasks to be used, by default this will be inferred from the Schema.
    embedding_option_kwargs: Optional[dict]
        Additional arguments to provide to `EmbeddingOptions` object for embeddings tables setting.
        Defaults to {}

    Returns
    -------
    Model

    """

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
    
    if schema is None:
        raise ValueError("The schema is required by Wide and Deep Model")

    if embedding_dim is None:
        raise ValueError("The embedding_dim is required")

    if not wide_schema:
        warnings.warn(
            f"""If not specify wide_schema, all features would be sent to wide
                model"""
        )
        wide_schema = schema

    if not deep_schema:
        deep_schema = schema
    if len(deep_schema) > 0 and not deep_block:
        raise ValueError(
            "The deep_block is required by Deep & Wide Model when "
            "features are available in the deep_schema"
        )

    if len(deep_schema) > 0:
        if not deep_input_block:
            deep_input_block = InputBlock(
                deep_schema,
                embedding_options=EmbeddingOptions(
                    embedding_dim_default=embedding_dim, **embedding_option_kwargs
                ),
                **kwargs
            )
        deep_body = deep_input_block.connect(deep_block).connect(MLPBlock(dimensions=[1], no_activation_last_layer=True))

    if len(wide_schema) > 0:
        if not wide_input_block:
            wide_input_block = InputBlock(
                wide_schema,
                embedding_options=EmbeddingOptions(
                    embedding_dim_default=embedding_dim, **embedding_option_kwargs
                ),
                **kwargs
            )
        wide_body = wide_input_block.connect(MLPBlock(dimensions=[1], no_activation_last_layer=True))

    branches = {
        "wide": wide_body,
        "deep": deep_body
    }
    wide_and_deep_body = ParallelBlock(branches, aggregation="element-wise-sum")
    model = Model(wide_and_deep_body, prediction_tasks)

    return model