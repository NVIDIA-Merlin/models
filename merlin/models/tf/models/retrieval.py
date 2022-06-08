from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from merlin.models.tf.blocks.core.aggregation import SequenceAggregation, SequenceAggregator
from merlin.models.tf.blocks.core.base import Block, BlockType
from merlin.models.tf.blocks.mlp import MLPBlock
from merlin.models.tf.blocks.retrieval.matrix_factorization import QueryItemIdsEmbeddingsBlock
from merlin.models.tf.blocks.retrieval.two_tower import TwoTowerBlock
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.inputs.embedding import EmbeddingOptions
from merlin.models.tf.models.base import Model, RetrievalModel
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.next_item import NextItemPredictionTask
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.schema import Schema, Tags


def MatrixFactorizationModel(
    schema: Schema,
    dim: int,
    query_id_tag=Tags.USER_ID,
    item_id_tag=Tags.ITEM_ID,
    embeddings_initializers: Optional[
        Union[Dict[str, Callable[[Any], None]], Callable[[Any], None]]
    ] = None,
    embeddings_l2_reg: float = 0.0,
    post: Optional[BlockType] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    logits_temperature: float = 1.0,
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> RetrievalModel:
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
    embeddings_initializers : Optional[Dict[str, Callable[[Any], None]]] = None
        An initializer function or a dict where keys are feature names and values are
        callable to initialize embedding tables
    embeddings_l2_reg: float = 0.0
        Factor for L2 regularization of the embeddings vectors (from the current batch only)
    post: Optional[Block], optional
        The optional `Block` to apply on both outputs of Two-tower model
    prediction_tasks: optional
        The optional `PredictionTask` or list of `PredictionTask` to apply on the model.
    logits_temperature: float
        Parameter used to reduce model overconfidence, so that logits / T.
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `bpr`.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`

    Returns
    -------
    RetrievalModel
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            logits_temperature=logits_temperature,
            samplers=list(samplers),
            **kwargs,
        )

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
    mf = QueryItemIdsEmbeddingsBlock(
        schema=schema,
        dim=dim,
        query_id_tag=query_id_tag,
        item_id_tag=item_id_tag,
        embeddings_initializers=embeddings_initializers,
        embeddings_l2_reg=embeddings_l2_reg,
        post=post,
        **kwargs,
    )

    model = RetrievalModel(mf, prediction_tasks)

    return model


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
    samplers: Sequence[ItemSampler] = (),
    **kwargs,
) -> RetrievalModel:
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
    logits_temperature: float
        Parameter used to reduce model overconfidence, so that logits / T.
        Defaults to 1.
    loss: Optional[LossType]
        Loss function.
        Defaults to `categorical_crossentropy`.
    samplers: List[ItemSampler]
        List of samplers for negative sampling, by default `[InBatchSampler()]`

    Returns
    -------
    RetrievalModel
    """

    if not prediction_tasks:
        prediction_tasks = ItemRetrievalTask(
            schema,
            logits_temperature=logits_temperature,
            samplers=list(samplers),
            **kwargs,
        )

    prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)
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

    model = RetrievalModel(two_tower, prediction_tasks)

    return model


def YoutubeDNNRetrievalModel(
    schema: Schema,
    max_seq_length: int,
    aggregation: str = "concat",
    top_block: Block = MLPBlock([64]),
    l2_normalization: bool = True,
    extra_pre_call: Optional[Block] = None,
    task_block: Optional[Block] = None,
    logits_temperature: float = 1.0,
    seq_aggregator: Block = SequenceAggregator(SequenceAggregation.MASKED_MEAN),
    sampled_softmax: bool = True,
    num_sampled: int = 100,
    min_sampled_id: int = 0,
    embedding_options: EmbeddingOptions = EmbeddingOptions(
        embedding_dims=None,
        embedding_dim_default=64,
        infer_embedding_sizes=False,
        infer_embedding_sizes_multiplier=2.0,
    ),
) -> Model:
    """Build the Youtube-DNN retrieval model.
    More details of the architecture can be found in [1]_.
    The sampled_softmax is enabled by default [2]_ [3]_ [4]_.

    Example Usage::
        model = YoutubeDNNRetrievalModel(schema, num_sampled=100)
        model.compile(optimizer="adam")
        model.fit(train_data, epochs=10)

    References
    ----------
    .. [1] Covington, Paul, Jay Adams, and Emre Sargin.
        "Deep neural networks for youtube recommendations."
        Proceedings of the 10th ACM conference on recommender systems. 2016.

    .. [2] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic
       Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial
       Intelligence and Statistics (AISTATS).

    .. [3] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312

    .. [4] Jean, Sébastien, et al. "On using very large target vocabulary for neural
        machine translation." arXiv preprint arXiv:1412.2007 (2014).


    Parameters
    ----------
    schema: Schema
        The `Schema` with the input features
    aggregation: str
        The aggregation method to use for the sequence of features.
        Defaults to `concat`.
    top_block: Block
        The `Block` that combines the top features
    l2_normalization: bool
        Whether to apply L2 normalization before computing dot interactions.
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
    sampled_softmax: bool
        Compute the logits scores over all items of the catalog or
        generate a subset of candidates
        Defaults to False
    num_sampled: int
        When sampled_softmax is enabled, specify the number of
        negative candidates to generate for each batch.
        Defaults to 100
    min_sampled_id: int
        The minimum id value to be sampled with sampled softmax.
        Useful to ignore the first categorical
        encoded ids, which are usually reserved for <nulls>,
        out-of-vocabulary or padding. Defaults to 0.
    embedding_options : EmbeddingOptions, optional
        An EmbeddingOptions instance, which allows for a number of
        options for the embedding table, by default EmbeddingOptions()
    """

    inputs = InputBlock(
        schema,
        aggregation=aggregation,
        seq=False,
        max_seq_length=max_seq_length,
        masking="clm",
        split_sparse=True,
        seq_aggregator=seq_aggregator,
        embedding_options=embedding_options,
    )

    task = NextItemPredictionTask(
        schema=schema,
        masking=True,
        weight_tying=False,
        extra_pre_call=extra_pre_call,
        task_block=task_block,
        logits_temperature=logits_temperature,
        l2_normalization=l2_normalization,
        sampled_softmax=sampled_softmax,
        num_sampled=num_sampled,
        min_sampled_id=min_sampled_id,
    )

    # TODO: Figure out how to make this fit as
    # a RetrievalModel (which must have a RetrievalBlock)
    return Model(inputs, top_block, task)
