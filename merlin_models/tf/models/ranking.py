from typing import List, Optional, Union

from merlin_standard_lib import Schema

from ..blocks.cross import CrossBlock
from ..blocks.dlrm import DLRMBlock
from ..blocks.inputs import InputBlock
from ..blocks.mlp import MLPBlock
from ..core import Block, Model, ParallelPredictionBlock, PredictionTask
from .utils import _parse_prediction_tasks


def DLRMModel(
    schema: Schema,
    embedding_dim: int,
    bottom_block: Block = None,
    top_block: Optional[Block] = None,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
) -> Model:
    prediction_tasks = _parse_prediction_tasks(schema, prediction_tasks)

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
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
    **kwargs
) -> Model:
    aggregation = kwargs.pop("aggregation", "concat")
    input_block = input_block or InputBlock(schema, aggregation=aggregation, **kwargs)
    prediction_tasks = _parse_prediction_tasks(schema, prediction_tasks)
    if stacked:
        dcn_body = input_block.connect(CrossBlock(depth), deep_block)
    else:
        dcn_body = input_block.connect_branch(CrossBlock(depth), deep_block, aggregation="concat")

    model = dcn_body.connect(prediction_tasks)

    return model


def YoutubeDNNRankingModel(schema: Schema) -> Model:
    raise NotImplementedError()
