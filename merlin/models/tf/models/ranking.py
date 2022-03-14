from typing import List, Optional, Union

from merlin.models.tf.blocks.core.base import Block
from merlin.models.tf.blocks.core.inputs import InputBlock
from merlin.models.tf.blocks.cross import CrossBlock
from merlin.models.tf.blocks.dlrm import DLRMBlock
from merlin.models.tf.blocks.mlp import MLPBlock
from merlin.models.tf.features.embedding import EmbeddingOptions
from merlin.models.tf.models.base import Model
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.schema import Schema


def DLRMModel(
    schema: Schema,
    embedding_dim: int,
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
    embedding_dim : int
        Dimension of the embeddings
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
        bottom_block=bottom_block,
        top_block=top_block,
    )
    model = dlrm_body.connect(prediction_tasks)

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
    **kwargs
) -> Model:
    """Create a model using the archicture proposed in DCN V2: Improved Deep & Cross Network [1].
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

    model = dcn_body.connect(prediction_tasks)

    return model


def YoutubeDNNRankingModel(schema: Schema) -> Model:
    raise NotImplementedError()
