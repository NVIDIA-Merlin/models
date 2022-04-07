from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from merlin.models.tf.blocks.core.aggregation import SequenceAggregation, SequenceAggregator
from merlin.models.tf.blocks.core.base import Block, BlockType, MetricOrMetrics
from merlin.models.tf.blocks.core.inputs import InputBlock
from merlin.models.tf.blocks.mlp import MLPBlock
from merlin.models.tf.blocks.retrieval.matrix_factorization import QueryItemIdsEmbeddingsBlock
from merlin.models.tf.blocks.retrieval.two_tower import TwoTowerBlock
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.features.embedding import EmbeddingOptions
from merlin.models.tf.losses import LossType
from merlin.models.tf.metrics.ranking import ranking_metrics
from merlin.models.tf.models.base import Model, RetrievalModel
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.next_item import NextItemPredictionTask
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.schema import Schema, Tags


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
    logits_temperature: float = 1.0,
    loss: Optional[LossType] = "bpr",
    metrics: MetricOrMetrics = ItemRetrievalTask.DEFAULT_METRICS,
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> Union[Model, RetrievalModel]:
    """Builds a matrix factorization model.

    Matrix factorization is a simple embedding model. Given the feedback matrix A (m*n),
    where __m__ is the number of users (or queries) and __n__ is the number of items,
    the model learns:

    - A user embedding matrix (m*dim), where row i is the embedding for user i.
    - An item embedding matrix (n*dim), where row j is the embedding for item j.

    Example Usage::
        mf = MatrixFactorizationModel(schema, dim=128)
        mf.compile(optimizer="adam")
        mf.fit(train_data, epochs=10)

    TODO: Make sure it can link to the Schema & Tags
    TODO: Link to tutorial

    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features.
    dim: int
        The dimension of the embeddings.
        The optimal value of this parameter typically depends on your dataset size.
        Smaller values consume less memory and train more quickly.
        Larger values give the model more expressive power to model
        the interactions between users and items, and can produce a more accurate model,
        but at the cost of memory consumption and CPU/GPU usage.
    query_id_tag : Tag
        The tag to select the query feature, by default `Tags.USER`
    item_id_tag : Tag
        The tag to select the item feature, by default `Tags.ITEM`
    embeddings_initializers: Dict[str, Callable[[Any], None]]
        TODO: List different popular initializers, and how to provide them. Type-hint is wrong now.
        A dictionary of initializers for embeddings.
    post: Optional[Block], optional
        The optional `Block` to apply on both the user- and item-embedding.
    prediction_tasks: Optional[PredictionTask]
        The optional `PredictionTask` or list of `PredictionTask` to apply on the model.
        TODO: add default.
    logits_temperature: float
        Parameter used to reduce model overconfidence, so that logits / T.
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `bpr`.
        TODO: List out some other possible loss functions.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`
        TODO: call out that this is used in ItemRetrievalTask when no prediction_tasks are provided.

    Returns
    -------
    TODO: Explain in what instances either option is returned.
    Union[Model, RetrievalModel]
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            metrics=metrics,
            logits_temperature=logits_temperature,
            samplers=list(samplers),
            loss=loss,
            **kwargs,
        )

    two_tower = QueryItemIdsEmbeddingsBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        embeddings_initializers=embeddings_initializers,
        post=post,
        **kwargs,
    )

    model = two_tower.connect_prediction_tasks(schema, prediction_tasks)

    return model


def YoutubeDNNRetrievalModel(
    schema: Schema,
    max_seq_length: int,
    aggregation: str = "concat",
    top_block: Block = MLPBlock([64]),
    num_sampled: int = 100,
    loss: Optional[LossType] = "categorical_crossentropy",
    metrics=ranking_metrics(top_ks=[10]),
    normalize: bool = True,
    extra_pre_call: Optional[Block] = None,
    task_block: Optional[Block] = None,
    logits_temperature: float = 1.0,
    seq_aggregator: Block = SequenceAggregator(SequenceAggregation.MEAN),
) -> Model:
    """Build the Youtube-DNN retrieval model. More details of the model can be found in [1].

    This retrieval model consists of a session-based architecture that averages
    past user interactions embeddings. The model is then trained using
    a Next Item Prediction task with a Sampled Softmax layer.

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
        The `Schema` with the input features.
    aggregation: str
        The aggregation method to use for the sequence of features.
        Defaults to `concat`.
        You can choose a different aggregation method to group the features embeddings
        in one interaction tensor.
    top_block: Block
        The `Block` instance to apply on the top of the user's dense vector returned
        by the input block.
        Defaults to `MLPBlock([64])`.
    num_sampled: int
        The number of negative samples to use in the sampled-softmax.
        Defaults to 100.
        The larger the number, the more accurate the model will be, but
        the more memory/time it will use.
        You generally want to fine-tune this parameter to ensure
        the best trade-off between the model's accuracy
        and training time.
    loss: Optional[LossType]
        Loss function.
        Defaults to `categorical_crossentropy`.
    metrics: List[Metric]
        List of metrics to use.The recommended metrics are `ranking_metrics`.
        (e.g. RecallAt(k=10), NDCGAt(k=10))
        Defaults to `ranking_metrics(top_ks=[10])`
    normalize: bool
        Whether to normalize the embeddings.
        Defaults to True.
    extra_pre_call: Optional[Block]
        The optional `Block` to apply before the model.
    task_block: Optional[Block]
        The optional `Block` to apply on the model.
    logits_temperature: float
        Parameter used to reduce model overconfidence, so that logits / T.
        Defaults to 1.
    seq_aggregator: Block
        The `Block` to aggregate the sequence of features.
        You can choose between `MEAN`, `SUM`, `MAX`, `MIN`
        to build the user's embedding vector from the sequence of past interactions.
        Defaults to `SequenceAggregator(SequenceAggregation.MEAN)`.
    """

    inputs = InputBlock(
        schema,
        aggregation=aggregation,
        seq=False,
        max_seq_length=max_seq_length,
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
        logits_temperature=logits_temperature,
        normalize=normalize,
        num_sampled=num_sampled,
    )

    return inputs.connect(top_block, task)


def TwoTowerModel(
    schema: Schema,
    query_tower: Block,
    item_tower: Optional[Block] = None,
    query_tower_tag=Tags.USER,
    item_tower_tag=Tags.ITEM,
    embedding_options: EmbeddingOptions = EmbeddingOptions(
        embedding_dims=None,
        embedding_dim_default=64,
        infer_embedding_sizes=False,
        infer_embedding_sizes_multiplier=2.0,
    ),
    post: Optional[BlockType] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    logits_temperature: float = 1.0,
    loss: Optional[LossType] = "categorical_crossentropy",
    metrics: MetricOrMetrics = ItemRetrievalTask.DEFAULT_METRICS,
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> Union[Model, RetrievalModel]:
    """Builds the Two-tower architecture, as proposed in [1].
    You can think of this model as a matrix-factorization model with both user- and item-features.

    TODO: Add 3 sentence summary about this model. When would you want to use this?
    - Works better for larger datasets

    Example Usage::
        two_tower = TwoTowerModel(schema, MLPBlock([256, 64]))
        two_tower.compile(optimizer="adam")
        two_tower.fit(train_data, epochs=10)

    TODO: Link to tutorial

    References
    ----------
    [1] Yi, Xinyang, et al.
        "Sampling-bias-corrected neural modeling for large corpus item recommendations."
        Proceedings of the 13th ACM Conference on Recommender Systems. 2019.

    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features.
    query_tower: Block
        The `Block` that combines user features.
    item_tower: Optional[Block], optional
        The optional `Block` that combines items features.
        If not provided, a copy of the query_tower is used.
    query_tower_tag: Tag
        The tag to select query features, by default `Tags.USER`
    item_tower_tag: Tag
        The tag to select item features, by default `Tags.ITEM`
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
    post: Optional[Block], optional
        The optional `Block` to apply on both outputs of Two-tower model
    prediction_tasks: optional
        The optional `PredictionTask` or list of `PredictionTask` to apply on the model.
        TODO: Explain what happens when no prediction-taks is provided
    logits_temperature: float
        Parameter used to reduce model overconfidence, so that logits / T.
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `categorical_crossentropy`.
        TODO: List other options here.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`
        TODO: Explain this is only relevant when no prediction-taks is provided

    Returns
    -------
    TODO: Explain in what instances either option is returned.
    Union[Model, RetrievalModel]
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            metrics=metrics,
            logits_temperature=logits_temperature,
            samplers=list(samplers),
            loss=loss,
            # Two-tower outputs are already L2-normalized
            normalize=False,
            **kwargs,
        )

    two_tower = TwoTowerBlock(
        schema=schema,
        query_tower=query_tower,
        item_tower=item_tower,
        query_tower_tag=query_tower_tag,
        item_tower_tag=item_tower_tag,
        embedding_options=embedding_options,
        post=post,
        **kwargs,
    )

    model = two_tower.connect_prediction_tasks(schema, prediction_tasks)

    return model
