from typing import Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer

from merlin_standard_lib import Schema

from .. import PredictionTask
from ..block.multi_task import MMOE, CGCBlock
from ..core import Block, Head, InputBlock, TabularTransformationType


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
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        **kwargs
    ):
        task_names = [task.task_name for task in prediction_tasks]
        body = Block.parse_block(body).apply(
            MMOE(expert_block, num_experts=num_experts, output_names=task_names, gate_dim=gate_dim)
        )

        super().__init__(
            body,
            prediction_tasks,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            loss_reduction=loss_reduction,
            inputs=inputs,
            pre=pre,
            post=post,
            **kwargs,
        )

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

        super().__init__(
            Block.parse_block(body).apply(*cgc_blocks),
            prediction_tasks,
            task_blocks,
            task_weights,
            bias_block,
            loss_reduction,
            inputs,
            **kwargs,
        )

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
