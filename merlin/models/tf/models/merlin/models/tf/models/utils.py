from typing import List, Optional, Union

from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.multi import PredictionTasks
from merlin.schema import Schema


def parse_prediction_tasks(
    schema: Schema,
    prediction_tasks: Optional[
        Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
    ] = None,
) -> Union[PredictionTask, ParallelPredictionBlock]:
    if not prediction_tasks:
        prediction_tasks = PredictionTasks(schema)
    if isinstance(prediction_tasks, (list, tuple)):
        prediction_tasks = ParallelPredictionBlock(*prediction_tasks)

    return prediction_tasks
