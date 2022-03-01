from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from merlin.schema import Schema, Tags

from ..blocks.aggregation import SequenceAggregation, SequenceAggregator
from ..blocks.inputs import InputBlock
from ..blocks.mlp import MLPBlock
from ..blocks.retrieval import MatrixFactorizationBlock, TwoTowerBlock
from ..core import Block, BlockType, Model, ParallelPredictionBlock, PredictionTask
from ..losses import LossType
from ..metrics.ranking import ranking_metrics
from ..prediction.item_prediction import ItemRetrievalTask, ItemSampler, NextItemPredictionTask
from .utils import _parse_prediction_tasks


def MatrixFactorizationModel(
    schema: Schema,
    dim: int,
    query_id_tag=Tags.USER_ID,
    item_id_tag=Tags.ITEM_ID,
    embeddings_initializers: Optional[Dict[str, Callable[[Any], None]]] = None,
    post: Optional[BlockType] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    softmax_temperature: int = 1,
    loss: Optional[LossType] = "bpr",
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> Model:
    """Builds a matrix factorization model.

    Example Usage::
        mf = MatrixFactorizationModel(schema, dim=128)
        mf.compile(optimizer="adam")
        mf.fit(train_data, epochs=10)

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
    post: Optional[Block], optional
        The optional `Block` to apply on both outputs of Two-tower model
    prediction_tasks: optional
        The optional `PredictionTask` or list of `PredictionTask` to apply on the model.
    softmax_temperature: float
        Parameter used to reduce model overconfidence, so that softmax(logits / T).
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `bpr`.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`

    Returns
    -------
    Model
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            softmax_temperature=softmax_temperature,
            samplers=samplers,
            loss=loss,
            **kwargs,
        )

    prediction_tasks = _parse_prediction_tasks(schema, prediction_tasks)
    two_tower = MatrixFactorizationBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        embeddings_initializers=embeddings_initializers,
        post=post,
        **kwargs,
    )

    model = two_tower.connect(prediction_tasks)

    return model


def TwoTowerModel(
    schema: Schema,
    query_tower: Block,
    item_tower: Optional[Block] = None,
    query_tower_tag=Tags.USER,
    item_tower_tag=Tags.ITEM,
    embedding_dim_default: Optional[int] = 64,
    post: Optional[BlockType] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    softmax_temperature: int = 1,
    loss: Optional[LossType] = "categorical_crossentropy",
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> Model:
    """Builds the Two-tower architecture, as proposed in [1].

    Example Usage::
        two_tower = TwoTowerModel(schema, MLPBlock([256, 64]))
        two_tower.compile(optimizer="adam")
        two_tower.fit(train_data, epochs=10)

    References
    ----------
    [1] Yi, Xinyang, et al.
        "Sampling-bias-corrected neural modeling for large corpus item recommendations."
        Proceedings of the 13th ACM Conference on Recommender Systems. 2019.

    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features
    query_tower: Block
        The `Block` that combines user features
    item_tower: Optional[Block], optional
        The optional `Block` that combines items features.
        If not provided, a copy of the query_tower is used.
    query_tower_tag: Tag
        The tag to select query features, by default `Tags.USER`
    item_tower_tag: Tag
        The tag to select item features, by default `Tags.ITEM`
    embedding_dim_default: Optional[int], optional
        Dimension of the embeddings, by default 64
    post: Optional[Block], optional
        The optional `Block` to apply on both outputs of Two-tower model
    prediction_tasks: optional
        The optional `PredictionTask` or list of `PredictionTask` to apply on the model.
    softmax_temperature: float
        Parameter used to reduce model overconfidence, so that softmax(logits / T).
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `categorical_crossentropy`.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`

    Returns
    -------
    Model
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            softmax_temperature=softmax_temperature,
            samplers=samplers,
            loss=loss,
            **kwargs,
        )

    prediction_tasks = _parse_prediction_tasks(schema, prediction_tasks)
    two_tower = TwoTowerBlock(
        schema=schema,
        query_tower=query_tower,
        item_tower=item_tower,
        query_tower_tag=query_tower_tag,
        item_tower_tag=item_tower_tag,
        embedding_dim_default=embedding_dim_default,
        post=post,
        **kwargs,
    )

    model = two_tower.connect(prediction_tasks)

    return model


def YoutubeDNNRetrievalModel(
    schema: Schema,
    aggregation: str = "concat",
    top_block: Block = MLPBlock([64]),
    num_sampled: int = 100,
    loss: Optional[LossType] = "categorical_crossentropy",
    metrics=ranking_metrics(top_ks=[10, 20]),
    normalize: bool = True,
    extra_pre_call: Optional[Block] = None,
    task_block: Optional[Block] = None,
    softmax_temperature: float = 1,
    seq_aggregator: Block = SequenceAggregator(SequenceAggregation.MEAN),
) -> Model:
    """Build the Youtube-DNN retrieval model. More details of the model can be found in [1].

    Example Usage::
        model = YoutubeDNNRetrievalModel(schema, num_sampled=100)
        model.compile(optimizer="adam")
        model.fit(train_data, epochs=10)

    References
    ----------
    [1] Covington, Paul, Jay Adams, and Emre Sargin.
        "Deep neural networks for youtube recommendations."
        Proceedings of the 10th ACM conference on recommender systems. 2016.


    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features
    aggregation: str
        The aggregation method to use for the sequence of features.
        Defaults to `concat`.
    top_block: Block
        The `Block` that combines the top features
    num_sampled: int
        The number of negative samples to use in the sampled-softmax.
        Defaults to 100.
    loss: Optional[LossType]
        Loss function.
        Defaults to `categorical_crossentropy`.
    metrics: List[Metric]
        List of metrics to use.
        Defaults to `ranking_metrics(top_ks=[10, 20])`
    normalize: bool
        Whether to normalize the embeddings.
        Defaults to True.
    extra_pre_call: Optional[Block]
        The optional `Block` to apply before the model.
    task_block: Optional[Block]
        The optional `Block` to apply on the model.
    softmax_temperature: float
        Parameter used to reduce model overconfidence, so that softmax(logits / T).
        Defaults to 1.
    seq_aggregator: Block
        The `Block` to aggregate the sequence of features.
    """

    inputs = InputBlock(
        schema,
        aggregation=aggregation,
        seq=False,
        masking="clm",
        split_sparse=True,
        seq_aggregator=seq_aggregator,
    )

    task = NextItemPredictionTask(
        schema=schema,
        loss=loss,
        metrics=metrics,
        masking=True,
        weight_tying=False,
        sampled_softmax=True,
        extra_pre_call=extra_pre_call,
        task_block=task_block,
        softmax_temperature=softmax_temperature,
        normalize=normalize,
        num_sampled=num_sampled,
    )

    return inputs.connect(top_block, task)
