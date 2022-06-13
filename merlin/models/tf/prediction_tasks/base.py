from typing import Dict, List, NamedTuple, Optional, Text, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import generic_utils

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    ContextMixin,
    ModelContext,
    PredictionOutput,
    name_fn,
)
from merlin.models.tf.blocks.core.combinators import ParallelBlock
from merlin.models.tf.typing import TabularData, TensorOrTabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils.schema_utils import tensorflow_metadata_json_to_schema
from merlin.schema import Schema, Tags


class TaskResults(NamedTuple):
    predictions: Union[TabularData, tf.Tensor]
    targets: Union[TabularData, tf.Tensor]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictionTask(Layer, ContextMixin):
    """Base-class for prediction tasks.

    Parameters
    ----------
    target_name : Optional[str], optional
        Label name, by default None
    task_name : Optional[str], optional
        Task name, by default None
    metrics : Optional[MetricOrMetrics], optional
        List of Keras metrics to be evaluated, by default None
    pre : Optional[Block], optional
        Optional block to transform predictions before computing loss and metrics,
        by default None
    pre_eval_topk : Optional[Block], optional
        Optional block to apply additional transform predictions before computing
        top-k evaluation loss and metrics, by default None
    task_block : Optional[Layer], optional
        Optional block to apply to inputs before computing predictions,
        by default None
    prediction_metrics : Optional[List[tf.keras.metrics.Metric]], optional
        List of Keras metrics used to summarize the predictions, by default None
    label_metrics : Optional[List[tf.keras.metrics.Metric]], optional
        List of Keras metrics used to summarize the labels, by default None
    compute_train_metrics : Optional[bool], optional
        Enable computing metrics during training, by default True
    name : Optional[Text], optional
        Task name, by default None
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        pre: Optional[Block] = None,
        pre_eval_topk: Optional[Block] = None,
        task_block: Optional[Layer] = None,
        name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.target_name = target_name
        self.task_block = task_block
        self._task_name = task_name
        self.pre = pre
        self._pre_eval_topk = pre_eval_topk

    @property
    def pre_eval_topk(self):
        return self._pre_eval_topk

    @pre_eval_topk.setter
    def pre_eval_topk(self, value: Block):
        """Set pre_eval_topk Block

        Parameters
        ----------
        value : Block
            The block for top-k evaluation
        """
        self._pre_eval_topk = value

    def pre_call(self, inputs: TensorOrTabularData, **kwargs) -> tf.Tensor:
        """Apply PredictionTask to inputs to get predictions scores

        Parameters
        ----------
        inputs : TensorOrTabularData
            inputs of the prediction task

        Returns
        -------
        tf.Tensor
        """
        x = inputs

        if self.task_block:
            x = self.task_block(x)

        if self.pre:
            x = self.pre(inputs, **kwargs)

        return x

    def pre_loss(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
        """Apply `call_outputs` method of `pre` block to transform predictions and targets
        before computing loss and metrics.

        Parameters
        ----------
        outputs : PredictionOutput
            The named tuple containing predictions and targets tensors

        Returns
        -------
        PredictionOutput
             The named tuple containing transformed predictions and targets tensors
        """
        out = self.pre.call_outputs(outputs, **kwargs) if self.pre else outputs

        return out

    def __call__(self, *args, **kwargs):
        inputs = self.pre_call(*args, **kwargs)

        # This will call the `call` method implemented by the super class.
        outputs = super().__call__(inputs, **kwargs)  # noqa

        if "targets" in kwargs:
            targets = kwargs.get("targets", {})
            if isinstance(targets, dict) and self.target_name:
                targets = targets[self.target_name]

            if isinstance(outputs, dict) and self.target_name and self.task_name in outputs:
                outputs = outputs[self.task_name]

            prediction_output = PredictionOutput(outputs, targets)

            if self.pre:
                prediction_output = self.pre_loss(prediction_output, **kwargs)

            if (
                isinstance(prediction_output.targets, tf.Tensor)
                and len(prediction_output.targets.shape)
                == len(prediction_output.predictions.shape) - 1
            ):
                prediction_output = prediction_output.copy_with_updates(
                    predictions=tf.squeeze(prediction_output.predictions)
                )

            return prediction_output

        return outputs

    def build_task(self, input_shape, schema: Schema, body: Block, **kwargs):
        return super().build(input_shape)

    @property
    def task_name(self):
        if self._task_name:
            return self._task_name

        base_name = generic_utils.to_snake_case(self.__class__.__name__)

        return name_fn(self.target_name, base_name) if self.target_name else base_name

    def child_name(self, name):
        return name_fn(self.task_name, name)

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            {
                "pre": tf.keras.layers.deserialize,
                "task_block": tf.keras.layers.deserialize,
            },
        )

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self,
            config,
            ["pre", "task_block"],
        )

        # config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ParallelPredictionBlock(ParallelBlock):
    """Multi-task prediction block.

    Parameters
    ----------
    prediction_tasks: *PredictionTask
        List of tasks to be used for prediction.
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]]
        Task blocks to be used for prediction.
    task_weights : Optional[List[float]]
        Weights for each task.
    bias_block : Optional[Layer]
        Bias block to be used for prediction.
    """

    def __init__(
        self,
        *prediction_tasks: PredictionTask,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        bias_block: Optional[Layer] = None,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        **kwargs,
    ):
        self.prediction_tasks = prediction_tasks
        self.bias_block = bias_block
        self.bias_logit = tf.keras.layers.Dense(1)

        self.prediction_task_dict = {}
        if prediction_tasks:
            for task in prediction_tasks:
                self.prediction_task_dict[task.task_name] = task

        super(ParallelPredictionBlock, self).__init__(self.prediction_task_dict, pre=pre, post=post)

        if task_blocks:
            self._set_task_blocks(task_blocks)

    @classmethod
    def get_tasks_from_schema(cls, schema, task_weight_dict=None):
        task_weight_dict = task_weight_dict or {}

        tasks: List[PredictionTask] = []
        task_weights = []
        from merlin.models.tf.prediction_tasks.classification import BinaryClassificationTask
        from merlin.models.tf.prediction_tasks.regression import RegressionTask

        for binary_target in schema.select_by_tag(Tags.BINARY_CLASSIFICATION).column_names:
            tasks.append(BinaryClassificationTask(binary_target))
            task_weights.append(task_weight_dict.get(binary_target, 1.0))
        for regression_target in schema.select_by_tag(Tags.REGRESSION).column_names:
            tasks.append(RegressionTask(regression_target))
            task_weights.append(task_weight_dict.get(regression_target, 1.0))
        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return task_weights, tasks

    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weight_dict: Optional[Dict[str, float]] = None,
        bias_block: Optional[Layer] = None,
        **kwargs,
    ) -> "ParallelPredictionBlock":
        """Built Multi-task prediction Block from schema

        Parameters
        ----------
        schema : Schema
            The `Schema` with the input features
        task_blocks : Optional[Union[Layer, Dict[str, Layer]]], optional
            Task blocks to be used for prediction, by default None
        task_weight_dict : Optional[Dict[str, float]], optional
            Weights for each task, by default None
        bias_block : Optional[Layer], optional
            Bias block to be used for prediction, by default None
        """
        task_weight_dict = task_weight_dict or {}

        task_weights, tasks = cls.get_tasks_from_schema(schema, task_weight_dict)

        return cls(
            *tasks,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            **kwargs,
        )

    @classmethod
    def task_names_from_schema(cls, schema: Schema) -> List[str]:
        _, tasks = cls.get_tasks_from_schema(schema)

        return [task.task_name for task in tasks]

    def _set_task_blocks(self, task_blocks):
        if isinstance(task_blocks, dict):
            tasks_multi_names = self._prediction_tasks_multi_names()
            for key, task_block in task_blocks.items():
                if key in tasks_multi_names:
                    tasks = tasks_multi_names[key]
                    if len(tasks) == 1:
                        self.prediction_task_dict[tasks[0].task_name].task_block = task_block
                    else:
                        raise ValueError(
                            f"Ambiguous name: {key}, can't resolve it to a task "
                            "because there are multiple tasks that contain the key: "
                            f"{', '.join([task.task_name for task in tasks])}"
                        )
                else:
                    raise ValueError(
                        f"Couldn't find {key} in prediction_tasks, "
                        f"only found: {', '.join(list(self.prediction_task_dict.keys()))}"
                    )
        elif isinstance(task_blocks, Layer):
            for key, val in self.prediction_task_dict.items():
                task_block = task_blocks.from_config(task_blocks.get_config())
                val.task_block = task_block
        else:
            raise ValueError("`task_blocks` must be a Layer or a Dict[str, Layer]")

    def _prediction_tasks_multi_names(self) -> Dict[str, List[PredictionTask]]:
        prediction_tasks_multi_names = {
            name: [val] for name, val in self.prediction_task_dict.items()
        }
        for name, value in self.prediction_task_dict.items():
            name_parts = name.split("/")
            for name_part in name_parts:
                if name_part in prediction_tasks_multi_names:
                    prediction_tasks_multi_names[name_part].append(value)
                else:
                    prediction_tasks_multi_names[name_part] = [value]

        return prediction_tasks_multi_names

    def add_task(self, task: PredictionTask, task_weight=1):
        key = task.target_name
        if not key:
            raise ValueError("PredictionTask must have a target_name")
        self.parallel_dict[key] = task
        if task_weight:
            self._task_weight_dict[key] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self.parallel_dict.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def call(
        self,
        inputs: Union[TabularData, tf.Tensor],
        training: bool = False,
        bias_outputs=None,
        **kwargs,
    ):
        if isinstance(inputs, dict) and not all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            if self.bias_block and not bias_outputs:
                bias_outputs = self.bias_block(inputs, training=training)
            inputs = self.body(inputs)

        outputs = super(ParallelPredictionBlock, self).call(inputs, training=training, **kwargs)

        if bias_outputs is not None:
            for key in outputs:
                outputs[key] += bias_outputs

        return outputs

    def compute_call_output_shape(self, input_shape):
        if isinstance(input_shape, dict) and not all(
            name in input_shape for name in list(self.parallel_dict.keys())
        ):
            input_shape = self.body.compute_output_shape(input_shape)

        return super().compute_call_output_shape(input_shape)

    @property
    def task_blocks(self) -> Dict[str, Optional[Layer]]:
        return {name: task.task_block for name, task in self.prediction_task_dict.items()}

    @property
    def task_names(self) -> List[str]:
        return [name for name in self.prediction_task_dict]

    def repr_ignore(self) -> List[str]:
        return ["prediction_tasks", "parallel_layers"]

    def _set_context(self, context: "ModelContext"):
        for task in self.prediction_task_dict.values():
            task._set_context(context)
        super(ParallelPredictionBlock, self)._set_context(context)

    @classmethod
    def from_config(cls, config, **kwargs):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["body", "prediction_tasks"])

        if "schema" in config:
            config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])

        prediction_tasks = config.pop("prediction_tasks", [])

        return cls(*prediction_tasks, **config)

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(self, config, ["body", "prediction_tasks"])

        return config
