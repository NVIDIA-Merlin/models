from collections import deque
from typing import Optional

import tensorflow as tf
from tensorflow.python.ops import array_ops

from ..core import Block, Sampler
from ..typing import TabularData


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class MemoryBankBlock(Block, Sampler):
    def __init__(
        self,
        num_batches: int = 1,
        key: Optional[str] = None,
        post: Optional[Block] = None,
        no_outputs: bool = False,
        stop_gradient: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.key = key
        self.num_batches = num_batches
        self.queue = deque(maxlen=num_batches + 1)
        self.no_outputs = no_outputs
        self.post = post
        self.stop_gradient = stop_gradient

    def call(self, inputs: TabularData, training=True, **kwargs) -> TabularData:
        if training:
            to_add = inputs[self.key] if self.key else inputs
            self.queue.append(to_add)

        if self.no_outputs:
            return {}

        return inputs

    def sample(self) -> tf.Tensor:
        outputs = tf.concat(list(self.queue)[:-1], axis=0)

        if self.post is not None:
            outputs = self.post(outputs)

        if self.stop_gradient:
            outputs = array_ops.stop_gradient(outputs, name="memory_bank_stop_gradient")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
