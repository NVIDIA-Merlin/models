import warnings
from typing import List, Optional, Union

import tensorflow as tf

from merlin.models.tf.blocks.cross import CrossBlock
from merlin.models.tf.blocks.dlrm import DLRMBlock
from merlin.models.tf.blocks.interaction import FMBlock
from merlin.models.tf.blocks.mlp import MLPBlock, RegularizerType
from merlin.models.tf.core.aggregation import ConcatFeatures
from merlin.models.tf.core.base import Block, BlockType
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.inputs.base import InputBlockV2
from merlin.models.tf.inputs.embedding import EmbeddingOptions, Embeddings
from merlin.models.tf.models.base import Model
from merlin.models.tf.models.utils import parse_prediction_blocks
from merlin.models.tf.outputs.base import ModelOutputType
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.transforms.features import CategoryEncoding
from merlin.schema import Schema, Tags


def DLRMModel(
    schema: Schema,
    *,
    embeddings: Optional[Block] = None,
    embedding_dim: Optional[int] = None,
    embedding_options: Optional[EmbeddingOptions] = None,
    bottom_block: Optional[Block] = None,
    top_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[
            PredictionTask,
            List[PredictionTask],
            ParallelPredictionBlock,
            ModelOutputType,
        ]
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
    schema : ~merlin.schema.Schema
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
    prediction_tasks: Optional[Union[PredictionTask,List[PredictionTask],
                                ParallelPredictionBlock,ModelOutputType]
        The prediction tasks to be used, by default this will be inferred from the Schema.
        For custom prediction tasks we recommending using OutputBlock and blocks based
        on ModelOutput than the ones based in PredictionTask (that will be deprecated).

    Returns
    -------
    Model

    """

    prediction_blocks = parse_prediction_blocks(schema, prediction_tasks)

    dlrm_body = DLRMBlock(
        schema,
        embedding_dim=embedding_dim,
        embedding_options=embedding_options,
        embeddings=embeddings,
        bottom_block=bottom_block,
        top_block=top_block,
    )
    model = Model(dlrm_body, prediction_blocks)

    return model


def DCNModel(
    schema: Schema,
    depth: int,
    deep_block: Block = MLPBlock([512, 256]),
    stacked=True,
    input_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[
            PredictionTask,
            List[PredictionTask],
            ParallelPredictionBlock,
            ModelOutputType,
        ]
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
    schema : ~merlin.schema.Schema
        The `Schema` with the input features
    depth : int, optional
        Number of cross-layers to be stacked, by default 1
    deep_block : Block, optional
        The `Block` to use as the deep part of the model (typically a `MLPBlock`)
    stacked : bool
        Whether to use the stacked version of the model or the parallel version.
    input_block : Block, optional
        The `Block` to use as the input layer. If None, a default `InputBlockV2` object
        is instantiated, that creates the embedding tables for the categorical features
        based on the schema. The embedding dimensions are inferred from the features
        cardinality. For a custom representation of input data you can instantiate
        and provide an `InputBlockV2` instance.
    prediction_tasks: Optional[Union[PredictionTask,List[PredictionTask],
                                ParallelPredictionBlock,ModelOutputType]
        The prediction tasks to be used, by default this will be inferred from the Schema.
        For custom prediction tasks we recommending using OutputBlock and blocks based
        on ModelOutput than the ones based in PredictionTask (that will be deprecated).

    Returns
    -------
    SequentialBlock
        A `SequentialBlock` with a number of stacked Cross layers

    Raises
    ------
    ValueError
        Number of cross layers (depth) should be positive
    """

    input_block = input_block or InputBlockV2(schema, **kwargs)
    prediction_blocks = parse_prediction_blocks(schema, prediction_tasks)
    if stacked:
        dcn_body = input_block.connect(CrossBlock(depth), deep_block)
    else:
        dcn_body = input_block.connect_branch(CrossBlock(depth), deep_block, aggregation="concat")

    model = Model(dcn_body, prediction_blocks)

    return model


def DeepFMModel(
    schema: Schema,
    embedding_dim: Optional[int] = None,
    deep_block: Optional[Block] = None,
    input_block: Optional[Block] = None,
    wide_input_block: Optional[Block] = None,
    wide_logit_block: Optional[Block] = None,
    deep_logit_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[
            PredictionTask,
            List[PredictionTask],
            ParallelPredictionBlock,
            ModelOutputType,
        ]
    ] = None,
    **kwargs,
) -> Model:
    """DeepFM-model architecture, which is the sum of the 1-dim output
    of a Factorization Machine [2] and a Deep Neural Network

    Example Usage::
        deep_fm = DeepFMModel(schema, embedding_dim=64, deep_block=MLPBlock([256, 64]))
        deep_fm.compile(optimizer="adam")
        deep_fm.fit(train_data, epochs=10)

    References
    ----------
    [1] Huifeng, Guo, et al.
        "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
        arXiv:1703.04247  (2017).
    [2] Steffen, Rendle, "Factorization Machines" IEEE International
    Conference on Data Mining, 2010. https://ieeexplore.ieee.org/document/5694074

    Parameters
    ----------
    schema : ~merlin.schema.Schema
        The `Schema` with the input features
    embedding_dim : int
        Dimension of the embeddings
    deep_block : Optional[Block]
        The `Block` that learns high-order feature interactions.
        On top of this Block, an MLPBlock([1]) is added to output
        a 1-dim logit.
        Defaults to MLPBlock([64])
    input_block : Block, optional
        The `Block` to use as the input layer for the FM and Deep components.
        If None, a default `InputBlockV2` object
        is instantiated, that creates the embedding tables for the categorical features
        based on the schema, with the specified embedding_dim.
        For a custom representation of input data you can instantiate
        and provide an `InputBlockV2` instance.
    wide_input_block: Optional[Block], by default None
        The input for the wide block. If not provided,
        creates a default block that encodes categorical features
        with one-hot / multi-hot representation and also includes the continuous features.
    wide_logit_block: Optional[Block], by default None
        The output layer of the wide input. The last dimension needs to be 1.
        You might want to provide your own output logit block if you want to add
        dropout or kernel regularization to the wide block.
    deep_logit_block: Optional[Block], by default MLPBlock([1], activation="linear", use_bias=True)
        The output layer of the deep block. The last dimension needs to be 1.
        You might want to provide your own output logit block if you want to add
        dropout or kernel regularization to the wide block.

    prediction_tasks: Optional[Union[PredictionTask,List[PredictionTask],
                                ParallelPredictionBlock,ModelOutputType]
        The prediction tasks to be used, by default this will be inferred from the Schema.
        For custom prediction tasks we recommending using OutputBlock and blocks based
        on ModelOutput than the ones based in PredictionTask (that will be deprecated).
    Returns
    -------
    Model

    """

    input_block = input_block or InputBlockV2(
        schema,
        aggregation=None,
        categorical=Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=embedding_dim),
    )

    fm_tower = FMBlock(
        schema,
        fm_input_block=input_block,
        wide_input_block=wide_input_block,
        wide_logit_block=wide_logit_block,
    )

    if deep_block is None:
        deep_block = MLPBlock([64])
    deep_block = deep_block.prepare(aggregation=ConcatFeatures())

    deep_logit_block = deep_logit_block or MLPBlock([1], activation="linear", use_bias=True)
    deep_tower = input_block.connect(deep_block).connect(deep_logit_block)

    deep_fm = ParallelBlock({"fm": fm_tower, "deep": deep_tower}, aggregation="element-wise-sum")

    prediction_blocks = parse_prediction_blocks(schema, prediction_tasks)
    model = Model(deep_fm, prediction_blocks)

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
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock, ModelOutputType]
    ] = None,
    pre: Optional[BlockType] = None,
    **wide_body_kwargs,
) -> Model:
    """
    The Wide&Deep architecture [1] was proposed by Google
    in 2016 to balance between the ability of neural networks to generalize and capacity
    of linear models to memorize relevant feature interactions. The deep part is an MLP
    model, with categorical features represented as embeddings, which are concatenated
    with continuous features and fed through multiple MLP layers. The wide part is a
    linear model takes a sparse representation of categorical features (i.e. one-hot
    or multi-hot representation). Both wide and deep sub-models output a logit,
    which is summed and followed by sigmoid for binary classification loss.

    Example Usage::

        1. Using default input block
        ```python
        wide_deep = ml.benchmark.WideAndDeepModel(
            schema,
            deep_block=ml.MLPBlock([32, 16]),
            wide_schema=wide_schema,
            deep_schema=deep_schema,
            prediction_tasks=ml.BinaryOutput("click"),
        )
        wide_deep.compile(optimizer="adam")
        wide_deep.fit(train_data, epochs=10)
        ```

        2. Custom input block
        ```python
        deep_embedding = ml.Embeddings(schema, embedding_dim_default=8, infer_embedding_sizes=False)
        model = ml.WideAndDeepModel(
            schema,
            deep_input_block = ml.InputBlockV2(schema=schema, categorical=deep_embedding),
            wide_schema=wide_schema,
            wide_preprocess=ml.CategoryEncoding(wide_schema, output_mode="multi_hot", sparse=True),
            deep_block=ml.MLPBlock([32, 16]),
            prediction_tasks=ml.BinaryOutput("click"),
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
            prediction_tasks=ml.BinaryOutput("click"),
        )
        ```

        4. Wide preprocess with multi-hot categorical features and hashed 2nd-level multi-hot
            feature interactions
        ```python

        one_hot_schema = schema.select_by_name(['categ_1', 'categ_2'])
        multi_hot_schema = schema.select_by_name(['categ_multi_hot_3'])
        wide_schema = one_hot_schema + multi_hot_schema

        # One-hot features
        one_hot_encoding = mm.SequentialBlock(
                   mm.Filter(one_hot_schema),
                   mm.CategoryEncoding(one_hot_schema, sparse=True, output_mode="one_hot"),
        )
        ```

        If your dataset contains multi-hot categorical features, i.e. features that may contain
        multiple categorical values for a data sample, you can instantiate the `AsDenseFeatures`
        block that converts the sparse representation of multi-hot features into a dense one
        (with maximum size defined) where the missing values are padded with zeros, as in the
        following example.

        ```python
        # Multi-hot features
        multi_hot_encoding = mm.SequentialBlock(
                mm.Filter(multi_hot_schema),
                ml.ToDense(multi_hot_schema),
                mm.CategoryEncoding(multi_hot_schema, sparse=True, output_mode="multi_hot")
        )
        ```
        Linear models are not able to compute feature interaction (like MLPs).
        So to give the wide part more power we perform paired feature interactions
        as a preprocessing step, so that every possible combination of the values of
        two categorical features is mapped to a single id. That way, the model is be
        able to pick paired feature relationships, e.g., a pattern between the a category
        of a product and the city of a user.
        Although, this approach leads to very high-cardinality resulting feature (product
        between the two features cardinalities). So typically we apply the hashing trick
        to limit the resulting cardinality. Below you can see how easily you can compute
        crossed features with Merlin Models.

        Note: some feature combinations might not add information to the model, for example
        the feature cross between the item id and item category, as every item only maps to a
        single item category. You can explicitly ignore those combinations to reduce a bit
        the feature space.

        ```python
        # 2nd-level features interaction
        features_crossing = mm.SequentialBlock(
                    mm.Filter(wide_schema),
                    ml.ToDense(wide_schema),
                    mm.HashedCrossAll(
                        wide_schema,
                        # The crossed features will be hashed to this number of bins
                        num_bins=100,
                        # Performs 2nd feature interactions, typically max is 3rd level
                        max_level=2,
                        output_mode="multi_hot",
                        sparse=True,
                        ignore_combinations=[["item_id", "item_category"],
                                            ["item_id", "item_brand"]]
                    ),
                )

        model = ml.WideAndDeepModel(
            schema,
            wide_schema=wide_schema,
            deep_schema=deep_schema,
            wide_preprocess=ml.ParallelBlock(
                [
                    one_hot_encoding,
                    multi_hot_encoding,
                    features_crossing
                ],
                aggregation="concat",
            ),
            deep_block=ml.MLPBlock([32, 16]),
            prediction_tasks=ml.BinaryOutput("click"),
        )
        ```

        5. On Wide&Deep paper [1] they proposed usage of separate optimizers for dense (AdaGrad) and
        sparse embeddings parameters (FTRL). You can implement that by using `MultiOptimizer` class.
        For example:
        ```python
            wide_model = model.blocks[0].parallel_layers["wide"]
            deep_model = model.blocks[0].parallel_layers["deep"]

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
    schema : ~merlin.schema.Schema
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
        Transformation block for preprocess data in wide model, such as CategoryEncoding,
        HashedCross, and HashedCrossAll. Please note the schema of transformation
        block should be the same as the wide_schema. See example usages.
        If wide_schema is provided and wide_preprocess, the CategoryEncoding transformation
        is used by default for one-hot encoding.
    deep_input_block : Optional[Block]
        The input block to be used by the deep part. It not provided, it is created internally
        by using the deep_schema. Defaults to None.
    wide_input_block : Optional[Block]
        The input block to be used by the wide part. It not provided, it is created internally
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
    prediction_tasks: Optional[Union[PredictionTask,List[PredictionTask],
                                ParallelPredictionBlock,ModelOutputType]
        The prediction tasks to be used, by default this will be inferred from the Schema.
        For custom prediction tasks we recommending using OutputBlock and blocks based
        on ModelOutput than the ones based in PredictionTask (that will be deprecated).

    Returns
    -------
    Model

    """

    prediction_blocks = parse_prediction_blocks(schema, prediction_tasks)

    if not wide_schema:
        warnings.warn("If not specify wide_schema, NO feature would be sent to wide model")

    if not deep_schema:
        deep_schema = schema

    branches = dict()

    if not deep_input_block:
        if deep_schema is not None and len(deep_schema) > 0:
            deep_input_block = InputBlockV2(
                deep_schema,
                categorical=Embeddings(
                    deep_schema.select_by_tag(Tags.CATEGORICAL),
                    sequence_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
                ),
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
            if wide_preprocess is None:
                wide_preprocess = (
                    CategoryEncoding(wide_schema, sparse=True, output_mode="one_hot"),
                )
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
                **wide_body_kwargs,
            )
        )
        branches["wide"] = wide_body

    if len(branches) == 0:
        raise ValueError(
            "At least the deep part (deep_schema/deep_input_block)"
            " or wide part (wide_schema/wide_input_block) must be provided."
        )

    wide_and_deep_body = ParallelBlock(branches, pre=pre, aggregation="element-wise-sum")
    model = Model(wide_and_deep_body, prediction_blocks)

    return model
