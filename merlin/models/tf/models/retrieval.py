from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import merlin.io
from merlin.models.tf.blocks.mlp import MLPBlock
from merlin.models.tf.blocks.retrieval.base import TowerBlock
from merlin.models.tf.blocks.retrieval.matrix_factorization import QueryItemIdsEmbeddingsBlock
from merlin.models.tf.blocks.retrieval.two_tower import TwoTowerBlock
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.core.base import Block, BlockType
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.inputs.embedding import EmbeddingOptions
from merlin.models.tf.models.base import ItemRecommenderModel, Model, RetrievalModel
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.next_item import NextItemPredictionTask
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.models.tf.predictions.dot_product import DotProductCategoricalPrediction
from merlin.models.tf.predictions.topk import TopKLayer, TopKPrediction
from merlin.models.utils.dataset import unique_rows_by_features
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
    aggregation: str = "concat",
    top_block: Block = MLPBlock([64]),
    l2_normalization: bool = True,
    extra_pre_call: Optional[Block] = None,
    task_block: Optional[Block] = None,
    logits_temperature: float = 1.0,
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
        embedding_options=embedding_options,
    )

    task = NextItemPredictionTask(
        schema=schema,
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


class TwoTowerModelV2(RetrievalModel):
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
    """

    def __init__(
        self,
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
        samplers: Sequence[ItemSampler] = ["in-batch"],
        **kwargs,
    ):
        if not prediction_tasks:
            prediction_tasks = DotProductCategoricalPrediction(
                schema,
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

        super().__init__(two_tower, prediction_tasks, **kwargs)

    def query_block(self) -> TowerBlock:
        return self.first._query_block

    def item_block(self) -> TowerBlock:
        return self.first._item_block

    def query_embeddings(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        query_tag: Union[str, Tags] = Tags.USER,
        query_id_tag: Union[str, Tags] = Tags.USER_ID,
    ) -> merlin.io.Dataset:
        """Export query embeddings from the model.
        Parameters
        ----------
        dataset : merlin.io.Dataset
            Dataset to export embeddings from.
        batch_size : int
            Batch size to use for embedding extraction.
        query_tag: Union[str, Tags], optional
            Tag to use for the query.
        query_id_tag: Union[str, Tags], optional
            Tag to use for the query id.
        Returns
        -------
        merlin.io.Dataset
            Dataset with the user/query features and the embeddings
            (one dim per column in the data frame)
        """
        from merlin.models.tf.utils.batch_utils import QueryEmbeddings

        get_user_emb = QueryEmbeddings(self, batch_size=batch_size)

        dataset = unique_rows_by_features(dataset, query_tag, query_id_tag).to_ddf()
        embeddings = dataset.map_partitions(get_user_emb)

        return merlin.io.Dataset(embeddings)

    def item_embeddings(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        item_tag: Union[str, Tags] = Tags.ITEM,
        item_id_tag: Union[str, Tags] = Tags.ITEM_ID,
        filter_input_columns: bool = False,
    ) -> merlin.io.Dataset:
        """Export item embeddings from the model.
        Parameters
        ----------
        dataset : merlin.io.Dataset
            Dataset to export embeddings from.
        batch_size : int
            Batch size to use for embedding extraction.
        item_tag : Union[str, Tags], optional
            Tag to use for the item.
        item_id_tag : Union[str, Tags], optional
            Tag to use for the item id, by default Tags.ITEM_ID
        filter_input_columns: bool
        Returns
        -------
        merlin.io.Dataset
            Dataset with the item features and the embeddings
            (one dim per column in the data frame)
        """
        from merlin.models.tf.utils.batch_utils import ItemEmbeddings

        get_item_emb = ItemEmbeddings(self, batch_size=batch_size)

        dataset = unique_rows_by_features(dataset, item_tag, item_id_tag).to_ddf()
        if filter_input_columns:
            id_column = self.schema.select_by_tag(item_id_tag).first.name
            embeddings = dataset.map_partitions(get_item_emb, filter_input_columns=[id_column])
        else:
            embeddings = dataset.map_partitions(get_item_emb)

        return merlin.io.Dataset(embeddings)

    def to_item_recommender(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        k: int = 10,
        prediction: TopKLayer = None,
        item_tag: Union[str, Tags] = Tags.ITEM,
        item_id_tag: Union[str, Tags] = Tags.ITEM_ID,
    ) -> ItemRecommenderModel:
        """Convert the retrieval model to a top-k recommender model
        for evaluation and inference

        Parameters
        ----------
        dataset : merlin.io.Dataset
            Dataset to export item embeddings from
        batch_size : int
            Batch size to use for embedding extraction
        k : int, optional
            Number of top candidates to retrieve, by default 10
        prediction : TopKLayer, optional
            The index layer for retrieving top-candidates, by default None
        item_tag : Union[str, Tags], optional
            Tag to use for the item, by default Tags.ITEM
        item_id_tag : Union[str, Tags], optional
            Tag to use for the item-id column, by default Tags.ITEM_ID

        Returns
        -------
        ItemRecommenderModel
            Top-k recommender model
        """
        item_embeddings = (
            self.item_embeddings(
                dataset,
                batch_size=batch_size,
                item_tag=item_tag,
                item_id_tag=item_id_tag,
                filter_input_columns=True,
            )
            .to_ddf()
            .compute()
        )
        id_column = self.schema.select_by_tag(item_id_tag).first.name
        item_embeddings.set_index(id_column, inplace=True)
        prediction = TopKPrediction(item_dataset=item_embeddings, prediction=prediction, k=k)
        return ItemRecommenderModel(self.query_block(), prediction)
