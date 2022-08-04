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
import warnings
from typing import Dict, List, Optional, Union

import merlin.models.tf as ml
from merlin.models.tf.blocks.retrieval.matrix_factorization import (
    MatrixFactorizationBlock,
    QueryItemIdsEmbeddingsBlock,
)
from merlin.models.tf.core.aggregation import ElementWiseMultiply
from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.inputs.embedding import EmbeddingOptions
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
    **kwargs,
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
    deep_block: Block,
    embedding_dims: Optional[Dict[str, int]] = None,
    embedding_dim_default: Optional[int] = None,
    wide_schema: Optional[Schema] = None,
    deep_schema: Optional[Schema] = None,
    wide_preprocess: Optional[Block] = None,
    deep_input_block: Optional[Block] = None,
    wide_input_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    embedding_option_kwargs: dict = {},
    **kwargs,
) -> Model:
    """Wide-and-Deep-model architecture.

    Example Usage::
        wide_deep = ml.benchmark.WideAndDeepModel(
                schema,
                deep_block=ml.MLPBlock([32, 16]),
                embedding_dims={"user_catetory": 32},
                embedding_dim_default=64,
                wide_schema=wide_schema,
                prediction_tasks=ml.BinaryClassificationTask("click"),
            )
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
    deep_block: Block
        Block (structure) of deep model.
    embedding_dims : Optional[dict]
        Dimensions of the embeddings, in order to specify different dimensions for different
        features, for features not specified by this dict, default dimension
        (embedding_dim_default) would be used to set their embeddings.by default None, then
        embedding_dim_default would be the fixed embedding dimension for all features
    embedding_dim : Optional[int]
        Default dimension of feateures not specified in embedding_dims
    wide_schema : Optional[Schema]
        The 'Schema' of input features for wide model, by default no features would be sent to
        wide model, and the model would become a pure deep model, if specified, only features
        in wide_schema would be sent to wide model
    deep_schema : Optional[Schema]
        The 'Schema' of input features for deep model, by default all features would be sent to
        deep model. deep_schema and wide_schema could contain the same features
    wide_preprocess : Optional[Block]
        Transformation block for preprocess data in wide model. Such as CategoricalOneHot,
        CategoryEncoding, HashedCross, and HashedCrossAll, please note the schema of transformation
        block should be the same as the wide_schema.  By default None.
        For example:
            ```python
            # CategoricalOneHot as preprocess for wide model
            import merlin.models.tf as ml
            model = ml.benchmark.WideAndDeepModel(
                schema = schema,
                wide_schema=wide_schema,
                deep_schema=deep_schema,
                wide_preprocess = ml.CategoricalOneHot(wide_schema)
                deep_block=ml.MLPBlock([32, 16]),
                prediction_tasks=ml.BinaryClassificationTask("click"),
            )

            # HashedCross as preprocess for wide model
            model = ml.benchmark.WideAndDeepModel(
                schema = schema,
                wide_schema=wide_schema,
                deep_schema=deep_schema,
                wide_preprocess = ml.HashedCross(wide_schema, num_bins=1000),
                deep_block=ml.MLPBlock([32, 16]),
                prediction_tasks=ml.BinaryClassificationTask("click"),
            )
            ```
    prediction_tasks: Optional[Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
        The prediction tasks to be used, by default this will be inferred from the Schema.
    embedding_option_kwargs: Optional[dict]
        Additional arguments to provide to `EmbeddingOptions` object for embeddings tables setting.
        Defaults to {}

    Returns
    -------
    Model

    """

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)

    if embedding_dims and embedding_dim_default:
        embedding_options = EmbeddingOptions(
            embedding_dims=embedding_dims,
            embedding_dim_default=embedding_dim_default,
            **embedding_option_kwargs,
        )
    elif embedding_dims:
        embedding_options = EmbeddingOptions(
            embedding_dims=embedding_dims, **embedding_option_kwargs
        )
    elif embedding_dim_default:
        embedding_options = EmbeddingOptions(
            embedding_dim_default=embedding_dim_default, **embedding_option_kwargs
        )
    else:
        embedding_options = EmbeddingOptions(**embedding_option_kwargs)

    if not wide_schema:
        warnings.warn("If not specify wide_schema, NO feature would be sent to wide model")
        wide_schema = None

    if not deep_schema:
        deep_schema = schema

    if not deep_input_block:
        if len(deep_schema) > 0:
            deep_input_block = InputBlock(
                deep_schema,
                embedding_options=embedding_options,
                **kwargs,
            )
    deep_body = deep_input_block.connect(deep_block).connect(
        ml.MLPBlock([1], no_activation_last_layer=True)
    )

    if not wide_input_block:
        if len(wide_schema) > 0:
            wide_input_block = ParallelBlock(
                TabularBlock.from_schema(schema=wide_schema, pre=wide_preprocess),
                is_input=True,
                aggregation="concat",
            )
    wide_body = wide_input_block.connect(ml.MLPBlock([1], no_activation_last_layer=True))

    branches = {"wide": wide_body, "deep": deep_body}
    wide_and_deep_body = ParallelBlock(branches, aggregation="element-wise-sum")
    model = Model(wide_and_deep_body, prediction_tasks)

    return model
