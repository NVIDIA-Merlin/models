import warnings
from typing import List, Optional, Union

from merlin.models.tf.core.combinators import ParallelBlock
from merlin.models.tf.outputs.base import ModelOutputType
from merlin.models.tf.outputs.block import ModelOutput, OutputBlock
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.multi import PredictionTasks
from merlin.schema import Schema


def parse_prediction_blocks(
    schema: Schema,
    prediction_blocks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock, ModelOutputType]
    ] = None,
) -> Union[PredictionTask, ParallelPredictionBlock, ModelOutputType]:
    if prediction_blocks is None:
        prediction_blocks = OutputBlock(schema)
    elif isinstance(prediction_blocks, (PredictionTask, ParallelPredictionBlock)) or (
        isinstance(prediction_blocks, (list, tuple))
        and isinstance(prediction_blocks[0], PredictionTask)
    ):
        prediction_blocks = parse_prediction_tasks(schema, prediction_blocks)
    elif not isinstance(prediction_blocks, (ModelOutput, ParallelBlock)):
        raise ValueError(
            "Invalid prediction blocks. It should be based either"
            "on ModelOutput (recommended) or PredictionTask (to be deprecated)."
        )

    return prediction_blocks


def parse_prediction_tasks(
    schema: Schema,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
) -> Union[PredictionTask, ParallelPredictionBlock]:
    warnings.warn(
        "PredictionTask based blocks are going to be deprecated."
        "Please move to ModelOutput based blocks (e.g. replace"
        "PredictionTasks() by OutputBlock().",
        DeprecationWarning,
    )

    if not prediction_tasks:
        prediction_tasks = PredictionTasks(schema)
    if isinstance(prediction_tasks, (list, tuple)):
        prediction_tasks = ParallelPredictionBlock(*prediction_tasks)
    elif not isinstance(prediction_tasks, (PredictionTask, ParallelPredictionBlock)):
        raise ValueError(
            "Invalid prediction task. If expects either a "
            "PredictionTask, List[PredictionTask], or ParallelPredictionBlock"
        )

    return prediction_tasks
