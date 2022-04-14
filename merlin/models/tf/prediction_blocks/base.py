from typing import TYPE_CHECKING, Optional

from tensorflow.keras.layers import Layer

from merlin.models.tf.blocks.core.base import Block, PredictionOutput

if TYPE_CHECKING:
    from merlin.models.tf.models.base import Model


class PreLossBlock(Block):
    def __init__(self, model: Optional[Model] = None, **kwargs):
        super().__init__(**kwargs)
        if model:
            self.model = model

    # OR pre_loss
    def pre_loss(self, inputs: PredictionOutput, training=False, **kwargs) -> PredictionOutput:
        return self.model.pre_loss()


# Or PreLossBlock
class PredictionBlock(Block):
    def __init__(self, model: Optional[Model] = None, **kwargs):
        super().__init__(**kwargs)
        if model:
            self.model = model

    # OR pre_loss
    def call(self, inputs: PredictionOutput, training=False, **kwargs) -> PredictionOutput:
        return inputs
