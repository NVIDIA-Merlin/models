from typing import Optional

from merlin_standard_lib import Schema

from ..blocks.aggregation import SequenceAggregation, SequenceAggregator
from ..blocks.inputs import InputBlock
from ..blocks.mlp import MLPBlock
from ..core import Block, Model
from ..losses import LossType
from ..metrics.ranking import ranking_metrics
from ..prediction.item_prediction import NextItemPredictionTask


def MatrixFactorizationModel() -> Model:
    pass


def TwoTowerModel() -> Model:
    pass


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
