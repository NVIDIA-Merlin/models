from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from merlin.models.tf.blocks.core.base import Block, PredictionOutput, TaskWithOutputs
from merlin.models.tf.typing import TabularData

if TYPE_CHECKING:
    from merlin.models.tf.models.base import Model


class PreLossBlock(Block):
    def call(
            self,
            features: TabularData,
            task_results: TaskWithOutputs,
            model: Model,
            training=False,
            testing=False,
    ) -> TaskWithOutputs:
        return task_results
