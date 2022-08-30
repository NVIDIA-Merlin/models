import warnings
from typing import List, Optional, Union

import tensorflow as tf

from merlin.models.tf.blocks.cross import CrossBlock
from merlin.models.tf.blocks.dlrm import DLRMBlock
from merlin.models.tf.blocks.interaction import FMPairwiseInteraction
from merlin.models.tf.blocks.mlp import MLPBlock, RegularizerType
from merlin.models.tf.core.aggregation import ConcatFeatures, StackFeatures
from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.core.transformations import CategoryEncoding
from merlin.models.tf.inputs.base import InputBlock, InputBlockV2
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import EmbeddingOptions
from merlin.models.tf.models.base import Model
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.schema import Schema


def DLRMModel(
    schema: Schema,
    *,
    embeddings: Optional[Block] = None,
    embedding_dim: Optional[int] = None,
    embedding_options: Optional[EmbeddingOptions] = None,
    bottom_block: Optional[Block] = None,
    top_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
) -> Model:
    """DLRM-model architecture.

    Example Usage::
        dlrm = DLRMModel(schema, embedding_dim=64, bottom_block=MLPBlock([256, 64]))
        dlrm.compile(optimizer="adam")
        dlrm.fit(train_data, epochs=10)

    References
    ----------
    [1] Naumov, Maxim, et al. "Deep learning recommendation model for
       personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019).

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    embeddings : Optional[Block]
        Optional block for categorical embeddings.
        Overrides the default embeddings inferred from the schema.
    embedding_dim : int
        Dimension of the embeddings
    embedding_options : Optional[EmbeddingOptions]
        Configuration for categorical embeddings. Alternatively use the embeddings parameter.
    bottom_block : Block
        The `Block` that combines the continuous features (typically a `MLPBlock`)
    top_block : Optional[Block], optional
        The optional `Block` that combines the outputs of bottom layer and of
        the factorization machine layer, by default None
    prediction_tasks: optional
        The prediction tasks to be used, by default this will be inferred from the Schema.

    Returns
    -------
    Model

    """

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)

    dlrm_body = DLRMBlock(
        schema,
        embedding_dim=embedding_dim,
        embedding_options=embedding_options,
        embeddings=embeddings,
        bottom_block=bottom_block,
        top_block=top_block,
    )
    model = Model(dlrm_body, prediction_tasks)

    return model


def DCNModel(
    schema: Schema,
    depth: int,
    deep_block: Block = MLPBlock([512, 256]),
    stacked=True,
    input_block: Optional[Block] = None,
    embedding_options: EmbeddingOptions = EmbeddingOptions(
        embedding_dims=None,
        embedding_dim_default=64,
        infer_embedding_sizes=False,
        infer_embedding_sizes_multiplier=2.0,
    ),
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    **kwargs,
) -> Model:
    """Create a model using the architecture proposed in DCN V2: Improved Deep & Cross Network [1].
    See Eq. (1) for full-rank and Eq. (2) for low-rank version.

    Example Usage::
        dcn = DCNModel(schema, depth=2, deep_block=MLPBlock([256, 64]))
        dcn.compile(optimizer="adam")
        dcn.fit(train_data, epochs=10)

    References
    ----------
    [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf


    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    depth : int, optional
        Number of cross-layers to be stacked, by default 1
    deep_block : Block, optional
        The `Block` to use as the deep part of the model (typically a `MLPBlock`)
    stacked : bool
        Whether to use the stacked version of the model or the parallel version.
    input_block : Block, optional
        The `Block` to use as the input layer, by default None
    embedding_options : EmbeddingOptions
        Options for the input embeddings.
        - embedding_dims: Optional[Dict[str, int]] - The dimension of the
        embedding table for each feature (key), by default {}
        - embedding_dim_default: int - Default dimension of the embedding
        table, when the feature is not found in ``embedding_dims``, by default 64
        - infer_embedding_sizes : bool, Automatically defines the embedding
        dimension from the feature cardinality in the schema, by default False
        - infer_embedding_sizes_multiplier: int. Multiplier used by the heuristic
        to infer the embedding dimension from its cardinality. Generally
        reasonable values range between 2.0 and 10.0. By default 2.0.
    prediction_tasks: optional
        The prediction tasks to be used, by default this will be inferred from the Schema.

    Returns
    -------
    SequentialBlock
        A `SequentialBlock` with a number of stacked Cross layers

    Raises
    ------
    ValueError
        Number of cross layers (depth) should be positive
    """

    aggregation = kwargs.pop("aggregation", "concat")
    input_block = input_block or InputBlock(
        schema, aggregation=aggregation, embedding_options=embedding_options, **kwargs
    )
    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
    if stacked:
        dcn_body = input_block.connect(CrossBlock(depth), deep_block)
    else:
        dcn_body = input_block.connect_branch(CrossBlock(depth), deep_block, aggregation="concat")

    model = Model(dcn_body, prediction_tasks)

    return model


def YoutubeDNNRankingModel(schema: Schema) -> Model:
    raise NotImplementedError()


def DeepFMModel(
    schema: Schema,
    embedding_dim: int,
    deep_block: Optional[Block] = MLPBlock([64, 128]),
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    embedding_option_kwargs: dict = {},
    **kwargs,
) -> Model:
    """DeepFM-model architecture.

    Example Usage::
        depp_fm = DeepFMModel(schema, embedding_dim=64, deep_block=MLPBlock([256, 64]))
        depp_fm.compile(optimizer="adam")
        depp_fm.fit(train_data, epochs=10)

    References
    ----------
    [1] Huifeng, Guo, et al.
        "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
        arXiv:1703.04247  (2017).

    Parameters
    ----------
    schema : Schema
        The `Schema` with the input features
    embedding_dim : int
        Dimension of the embeddings
    deep_block : Optional[Block]
        The `Block` that learns high-ordeer feature interactions
        Defaults to MLPBlock([64, 128])
    prediction_tasks: optional
        The prediction tasks to be used, by default this will be inferred from the Schema.
        Defaults to None
    embedding_option_kwargs: Optional[dict]
        Additional arguments to provide to `EmbeddingOptions` object
        for embeddings tables setting.
        Defaults to {}
    Returns
    -------
    Model

    """
    input_block = InputBlock(
        schema,
        embedding_options=EmbeddingOptions(
            embedding_dim_default=embedding_dim, **embedding_option_kwargs
        ),
        **kwargs,
    )

    pairwise_block = FMPairwiseInteraction().prepare(aggregation=StackFeatures(axis=-1))
    deep_block = deep_block.prepare(aggregation=ConcatFeatures())

    branches = {
        "categorical": CategoryEncoding(schema),
        "continuous": ContinuousFeatures.from_schema(schema),
    }
    first_order_block = ParallelBlock(branches, aggregation="concat").connect(
        tf.keras.layers.Dense(units=1, activation=None, use_bias=True)
    )

    deep_pairwise = input_block.connect_branch(pairwise_block, deep_block, aggregation="concat")
    deep_fm = ParallelBlock(
        {"deep_pairwise": deep_pairwise, "first_order": first_order_block}, aggregation="concat"
    )

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
    model = Model(deep_fm, prediction_tasks)

    return model


def WideAndDeepModel(
    schema: Schema,
    deep_block: Block,
    wide_schema: Optional[Schema] = None,
    deep_schema: Optional[Schema] = None,
    wide_preprocess: Optional[Block] = None,
    deep_input_block: Optional[Block] = None,
    wide_input_block: Optional[Block] = None,
    deep_regularizer: Optional[RegularizerType] = None,
    wide_regularizer: Optional[RegularizerType] = None,
    deep_dropout: Optional[float] = None,
    wide_dropout: Optional[float] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    **kwargs,
) -> Model:
    """Wide-and-Deep-model architecture [1].

    Example Usage::

        1. Using default input block
        ```python
        wide_deep = ml.benchmark.WideAndDeepModel(
            schema,
            deep_block=ml.MLPBlock([32, 16]),
            wide_schema=wide_schema,
            deep_schema=deep_schema,
            prediction_tasks=ml.BinaryClassificationTask("click"),
        )
        wide_deep.compile(optimizer="adam")
        wide_deep.fit(train_data, epochs=10)
        ```

        2. Custom input block
        ```python
        deep_embedding = ml.Embeddings(schema, embedding_dim_default=8, infer_embedding_sizes=False)
        model = ml.WideAndDeepModel(
            schema,
            deep_input_block = ml.InputBlockV2(schema=schema, embeddings=deep_embedding),
            wide_schema=wide_schema,
            wide_preprocess=ml.CategoryEncoding(wide_schema, output_mode="multi_hot", sparse=True),
            deep_block=ml.MLPBlock([32, 16]),
            prediction_tasks=ml.BinaryClassificationTask("click"),
        )
        ```

        3. Wide preprocess with one-hot categorical features and hashed 2nd-level feature
            interactions
        ```python
        model = ml.WideAndDeepModel(
            schema,
            wide_schema=wide_schema,
            deep_schema=deep_schema,
            wide_preprocess=ml.ParallelBlock(
                [
                    # One-hot representations of categorical features
                    ml.CategoryEncoding(wide_schema, output_mode="one_hot", sparse=True),
                    # One-hot representations of hashed 2nd-level feature interactions
                    ml.HashedCrossAll(wide_schema, num_bins=1000, max_level=2, sparse=True),
                ],
                aggregation="concat",
            ),
            deep_block=ml.MLPBlock([31, 16]),
            prediction_tasks=ml.BinaryClassificationTask("click"),
        )
        ```

        On Wide&Deep paper [1] they proposed usage of separate optimizers for dense (AdaGrad) and
        sparse embeddings parameters (FTRL). You can implement that by using `MultiOptimizer` class.
        For example:
        ```python
            wide_model = model.get_blocks_by_name("sequential_block_6")
            deep_model = model.get_blocks_by_name("sequential_block_3")

            multi_optimizer = ml.MultiOptimizer(
                default_optimizer="adagrad",
                optimizers_and_blocks=[
                    ml.OptimizerBlocks("ftrl", wide_model),
                    ml.OptimizerBlocks("adagrad", deep_model),
                ],
            )
        ```

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
    wide_schema : Optional[Schema]
        The 'Schema' of input features for wide model, by default no features would be sent to
        wide model, and the model would become a pure deep model, if specified, only features
        in wide_schema would be sent to wide model
    deep_schema : Optional[Schema]
        The 'Schema' of input features for deep model, by default all features would be sent to
        deep model. deep_schema and wide_schema could contain the same features
    wide_preprocess : Optional[Block]
        Transformation block for preprocess data in wide model. Such as CategoryEncoding,
        HashedCross, and HashedCrossAll, please note the schema of transformation
        block should be the same as the wide_schema. See example usage. By default None.
    deep_input_block : Optional[Block]
        The input block to be used by the deep part. It fnot provided, it is created internally
        by using the deep_schema. Defaults to None.
    wide_input_block : Optional[Block]
        The input block to be used by the wide part. It fnot provided, it is created internally
        by using the wide_schema. Defaults to None.
    deep_regularizer : Optional[RegularizerType]
        Regularizer function applied to the last layer kernel weights matrix and biases of
        the MLP layer of the wide part. Defaults to None.
    wide_regularizer : Optional[RegularizerType]
        Regularizer function applied to the last layer kernel weights matrix and biases
        of the last MLP layer of deep part). Defaults to None.
    deep_dropout: Optional[float]
        The dropout to be used by the last layer of deep part. Defaults to None.
    wide_dropout: Optional[float]
        The dropout to be used by the last layer of wide part. Defaults to None.
    prediction_tasks: Optional[Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
        The prediction tasks to be used, by default this will be inferred from the Schema.

    Returns
    -------
    Model

    """

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)

    if not wide_schema:
        warnings.warn("If not specify wide_schema, NO feature would be sent to wide model")

    if not deep_schema:
        deep_schema = schema

    branches = dict()

    if not deep_input_block:
        if deep_schema is not None and len(deep_schema) > 0:
            deep_input_block = InputBlockV2(
                deep_schema,
                **kwargs,
            )
    if deep_input_block:
        deep_body = deep_input_block.connect(deep_block).connect(
            MLPBlock(
                [1],
                no_activation_last_layer=True,
                kernel_regularizer=deep_regularizer,
                bias_regularizer=deep_regularizer,
                dropout=deep_dropout,
            )
        )
        branches["deep"] = deep_body

    if not wide_input_block:
        if wide_schema is not None and len(wide_schema) > 0:
            wide_input_block = ParallelBlock(
                TabularBlock.from_schema(schema=wide_schema, pre=wide_preprocess),
                is_input=True,
                aggregation="concat",
            )

    if wide_input_block:
        wide_body = wide_input_block.connect(
            MLPBlock(
                [1],
                no_activation_last_layer=True,
                kernel_regularizer=wide_regularizer,
                bias_regularizer=wide_regularizer,
                dropout=wide_dropout,
            )
        )
        branches["wide"] = wide_body

    if len(branches) == 0:
        raise ValueError(
            "At least the deep part (deep_schema/deep_input_block) "
            "or wide part (wide_schema/wide_input_block) must be provided."
        )

    wide_and_deep_body = ParallelBlock(branches, aggregation="element-wise-sum")
    model = Model(wide_and_deep_body, prediction_tasks)

    return model
