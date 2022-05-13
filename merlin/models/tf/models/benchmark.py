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
from merlin.models.tf.models.base import Model
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.schema import Schema


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
