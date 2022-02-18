from typing import List, Optional, Union

from merlin.schema import Schema

from ..blocks.multi_task import PredictionTasks
from ..core import ParallelPredictionBlock, PredictionTask


def _parse_prediction_tasks(
    schema: Schema,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
):
    if not prediction_tasks:
        prediction_tasks = PredictionTasks(schema)
    if isinstance(prediction_tasks, (list, tuple)):
        prediction_tasks = ParallelPredictionBlock(*prediction_tasks)

    return prediction_tasks
