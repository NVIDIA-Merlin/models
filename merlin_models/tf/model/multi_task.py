from typing import Dict, List, Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from tensorflow.python.keras.engine.base_layer import Layer

from .. import PredictionTask
from ..block.base import Block
from ..block.multi_task import MMOEGate, MultiExpertsBlock
from ..features.base import InputBlock
from .base import Head


class MMOEHead(Head):
    def __init__(
        self,
        body: tf.keras.layers.Layer,
        prediction_tasks: Union[List[PredictionTask], PredictionTask],
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_experts: int,
        gate_dim: int = 32,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        inputs: Optional[InputBlock] = None,
        **kwargs
    ):
        super().__init__(
            body,
            prediction_tasks,
            task_blocks,
            task_weights,
            bias_block,
            loss_reduction,
            inputs,
            **kwargs,
        )
        self.experts = MultiExpertsBlock(expert_block, num_experts)

        self.gate_dict: Dict[str, MMOEGate] = {}
        for task_name in self.prediction_task_dict:
            self.gate_dict[task_name] = MMOEGate(num_experts, dim=gate_dim)

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        body: Layer,
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_experts: int,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        inputs: Optional[InputBlock] = None,
        **kwargs
    ) -> "MMOEHead":
        task_weight_dict = task_weight_dict or {}

        tasks: List[PredictionTask] = []
        task_weights = []

        from .prediction_task import BinaryClassificationTask, RegressionTask

        for binary_target in schema.select_by_tag(Tag.BINARY_CLASSIFICATION).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))

        for regression_target in schema.select_by_tag(Tag.REGRESSION).column_names:
            tasks.append(RegressionTask(regression_target))
            task_weights.append(task_weight_dict.get(regression_target, 1.0))

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return cls(
            body,
            tasks,
            expert_block=expert_block,
            num_experts=num_experts,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            loss_reduction=loss_reduction,
            inputs=inputs,
            **kwargs,
        )

    def call_tasks(self, body_outputs, bias=None, **kwargs):
        outputs = {}

        expert_outputs = self.experts(body_outputs)

        for name, task in self.prediction_task_dict.items():
            task_inputs = self.gate_dict[name](body_outputs, expert_outputs)
            task_out = task(task_inputs, **kwargs)
            if bias is not None:
                task_out += bias
            outputs[name] = task_out

        return outputs
