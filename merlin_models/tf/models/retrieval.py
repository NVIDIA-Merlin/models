from typing import List, Optional, Union

from merlin_standard_lib import Schema, Tag

from ..blocks.aggregation import SequenceAggregation, SequenceAggregator
from ..blocks.inputs import InputBlock
from ..blocks.mlp import MLPBlock
from ..blocks.retrieval import TwoTowerBlock
from ..core import Block, BlockType, Model, ParallelPredictionBlock, PredictionTask
from ..losses import LossType
from ..metrics.ranking import ranking_metrics
from ..prediction.item_prediction import NextItemPredictionTask
from .utils import _parse_prediction_tasks


def MatrixFactorizationModel() -> Model:
    pass


def TwoTowerModel(
    schema: Schema,
    query_tower: Block,
    item_tower: Optional[Block] = None,
    query_tower_tag=Tag.USER,
    item_tower_tag=Tag.ITEM,
    embedding_dim_default: Optional[int] = 64,
    post: Optional[BlockType] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    **kwargs,
) -> Model:
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
    """
    Build the Youtube-DNN retrieval model.
    More details of the model can be found at
    [Covington et al., 2016](https://dl.acm.org/doi/10.1145/2959100.2959190Covington)
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
