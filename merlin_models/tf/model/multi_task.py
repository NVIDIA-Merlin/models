from typing import Dict, List, Optional, Union

import tensorflow as tf
from merlin_standard_lib import Schema
from tensorflow.python.keras.engine.base_layer import Layer

from .. import PredictionTask
from ..block.base import Block, SequentialBlock
from ..block.multi_task import CGCBlock, MMOEGate, MultiExpertsBlock
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

        task_weights, tasks = cls.get_tasks_from_schema(schema, task_weight_dict)

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


class PLEHead(Head):
    def __init__(
        self,
        body: tf.keras.layers.Layer,
        expert_block: Union[Block, tf.keras.layers.Layer],
        prediction_tasks: Union[List[PredictionTask], PredictionTask],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        depth: int = 1,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        inputs=None,
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
        cgc_blocks = []
        for i in range(depth):
            cgc_blocks.append(
                CGCBlock(
                    expert_block,
                    prediction_tasks,
                    num_task_experts,
                    num_shared_experts,
                    add_shared_gate=i < depth - 1,
                )
            )

        self.cgc = SequentialBlock(cgc_blocks)

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        body: Layer,
        expert_block: Union[Block, tf.keras.layers.Layer],
        num_task_experts: int = 1,
        num_shared_experts: int = 1,
        depth: int = 1,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        inputs: Optional[InputBlock] = None,
        **kwargs
    ) -> "PLEHead":
        task_weight_dict = task_weight_dict or {}

        task_weights, tasks = cls.get_tasks_from_schema(schema, task_weight_dict)

        return cls(
            body=body,
            expert_block=expert_block,
            prediction_tasks=tasks,
            num_task_experts=num_task_experts,
            num_shared_experts=num_shared_experts,
            depth=depth,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            loss_reduction=loss_reduction,
            inputs=inputs,
            **kwargs,
        )

    def call_tasks(self, body_outputs, bias=None, **kwargs):
        outputs = {}
        task_inputs = self.cgc(body_outputs)

        for name, task in self.prediction_task_dict.items():
            task_out = task(task_inputs[name], **kwargs)
            if bias is not None:
                task_out += bias
            outputs[name] = task_out

        return outputs
