import abc
from collections import defaultdict
from typing import Dict, List, Optional, Text, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import generic_utils

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockContext,
    BlockType,
    ContextMixin,
    MetricOrMetricClass,
    PredictionOutput,
    _output_metrics,
    name_fn,
)
from merlin.models.tf.blocks.core.combinators import ParallelBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.mixins import LossMixin, MetricsMixin
from merlin.models.utils.misc_utils import filter_kwargs
from merlin.models.utils.schema_utils import tensorflow_metadata_json_to_schema
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictionTask(Layer, LossMixin, MetricsMixin, ContextMixin):
    """Base-class for prediction tasks.

    Parameters
    ----------
    metrics:
        List of Keras metrics to be evaluated.
    prediction_metrics:
        List of Keras metrics used to summarize the predictions.
    label_metrics:
        List of Keras metrics used to summarize the labels.
    loss_metrics:
        List of Keras metrics used to summarize the loss.
    name:
        Optional task name.

    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        metrics: Optional[List[MetricOrMetricClass]] = None,
        pre: Optional[Block] = None,
        pre_metrics: Optional[Block] = None,
        task_block: Optional[Layer] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        compute_train_metrics: Optional[bool] = True,
        name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.target_name = target_name
        self.task_block = task_block
        self._task_name = task_name
        self.pre = pre
        self.pre_metrics = pre_metrics

        create_metrics = self._create_metrics
        self.eval_metrics = create_metrics(metrics) if metrics else []
        self.prediction_metrics = create_metrics(prediction_metrics) if prediction_metrics else []
        self.label_metrics = create_metrics(label_metrics) if label_metrics else []
        self.loss_metrics = create_metrics(loss_metrics) if loss_metrics else []
        self.compute_train_metrics = compute_train_metrics

    def pre_call(self, inputs, **kwargs):
        x = inputs

        if self.task_block:
            x = self.task_block(x)

        if self.pre:
            x = self.pre(inputs, **kwargs)

        return x

    def pre_loss(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
        out = self.pre.call_outputs(outputs, **kwargs) if self.pre else outputs

        return out

    def __call__(self, *args, **kwargs):
        inputs = self.pre_call(*args, **kwargs)

        # This will call the `call` method implemented by the super class.
        outputs = super().__call__(inputs, **kwargs)  # noqa

        return outputs

    def build_task(self, input_shape, schema: Schema, body: Block, **kwargs):
        return super().build(input_shape)

    def _create_metrics(self, metrics: List[MetricOrMetricClass]) -> List[tf.keras.metrics.Metric]:
        outputs = []
        for metric in metrics:
            if not isinstance(metric, tf.keras.metrics.Metric):
                metric = metric(name=self.child_name(generic_utils.to_snake_case(metric.__name__)))
            outputs.append(metric)

        return outputs

    @property
    def task_name(self):
        if self._task_name:
            return self._task_name

        base_name = generic_utils.to_snake_case(self.__class__.__name__)

        return name_fn(self.target_name, base_name) if self.target_name else base_name

    def child_name(self, name):
        return name_fn(self.task_name, name)

    @abc.abstractmethod
    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        raise NotImplementedError()

    def compute_loss(  # type: ignore
        self,
        predictions,
        targets,
        training: bool = False,
        compute_metrics=False,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        if isinstance(predictions, dict) and self.target_name and self.task_name in predictions:
            predictions = predictions[self.task_name]

        if self.pre:
            outputs = self.pre_loss(
                PredictionOutput(predictions, targets), training=training, **kwargs
            )
            targets, predictions = outputs.targets, outputs.predictions

        if isinstance(targets, tf.Tensor) and len(targets.shape) == len(predictions.shape) - 1:
            predictions = tf.squeeze(predictions)

        loss = self._compute_loss(
            predictions, targets=targets, sample_weight=sample_weight, training=training
        )

        if compute_metrics:
            if (not training) or (training and self.compute_train_metrics):
                update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

                update_ops = [x for x in update_ops if x is not None]

                with tf.control_dependencies(update_ops):
                    return tf.identity(loss)

        return loss

    def repr_add(self):
        return [("loss", self.loss)]

    def calculate_metrics(
        self, predictions, targets, sample_weight=None, forward=True, loss=None, **kwargs
    ):
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        if forward:
            predictions = self(predictions)

        if self.pre_metrics:
            outputs = self.pre_metrics.call_outputs(
                PredictionOutput(predictions, targets), **kwargs
            )
            targets, predictions = outputs.targets, outputs.predictions

        update_ops = []

        for metric in self.eval_metrics:
            update_ops.append(
                metric.update_state(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            )

        for metric in self.prediction_metrics:
            update_ops.append(metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self.label_metrics:
            update_ops.append(metric.update_state(targets, sample_weight=sample_weight))

        for metric in self.loss_metrics:
            if not loss:
                loss = self.loss(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
            update_ops.append(metric.update_state(loss, sample_weight=sample_weight))

        return update_ops

    def metric_results(self, mode: str = "val"):
        return {metric.name: metric.result() for metric in self.metrics}

    def metric_result_dict(self, mode=None):
        return self.metric_results(mode=mode)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            {
                "pre": tf.keras.layers.deserialize,
                "metrics": tf.keras.metrics.deserialize,
                "prediction_metrics": tf.keras.metrics.deserialize,
                "label_metrics": tf.keras.metrics.deserialize,
                "loss_metrics": tf.keras.metrics.deserialize,
            },
        )

        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self,
            config,
            ["metrics", "prediction_metrics", "label_metrics", "loss_metrics", "pre"],
        )

        # config["summary_type"] = self.sequence_summary.summary_type
        if self.target_name:
            config["target_name"] = self.target_name
        if self._task_name:
            config["task_name"] = self._task_name

        if "metrics" not in config:
            config["metrics"] = []

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ParallelPredictionBlock(ParallelBlock, LossMixin, MetricsMixin):
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
    loss_reduction : Callable
        Reduction function for loss.

    """

    def __init__(
        self,
        *prediction_tasks: PredictionTask,
        task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
        task_weights: Optional[List[float]] = None,
        bias_block: Optional[Layer] = None,
        loss_reduction=tf.reduce_mean,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        **kwargs,
    ):
        self.loss_reduction = loss_reduction

        self.prediction_tasks = prediction_tasks
        self.task_weights = task_weights

        self.bias_block = bias_block
        self.bias_logit = tf.keras.layers.Dense(1)

        self.prediction_task_dict = {}
        if prediction_tasks:
            for task in prediction_tasks:
                self.prediction_task_dict[task.task_name] = task

        super(ParallelPredictionBlock, self).__init__(self.prediction_task_dict, pre=pre, post=post)

        self._task_weight_dict = defaultdict(lambda: 1.0)
        if task_weights:
            for task, val in zip(prediction_tasks, task_weights):
                self._task_weight_dict[task.task_name] = val

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
        loss_reduction=tf.reduce_mean,
        **kwargs,
    ) -> "ParallelPredictionBlock":
        task_weight_dict = task_weight_dict or {}

        task_weights, tasks = cls.get_tasks_from_schema(schema, task_weight_dict)

        return cls(
            *tasks,
            task_blocks=task_blocks,
            task_weights=task_weights,
            bias_block=bias_block,
            loss_reduction=loss_reduction,
            **kwargs,
        )

    @classmethod
    def task_names_from_schema(cls, schema: Schema) -> List[str]:
        _, tasks = cls.get_tasks_from_schema(schema)

        return [task.task_name for task in tasks]

    def _set_task_blocks(self, task_blocks):
        if not task_blocks:
            return

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
                bias_outputs = self.bias_block(inputs)
            inputs = self.body(inputs)

        outputs = super(ParallelPredictionBlock, self).call(inputs, **kwargs)

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

    def compute_loss(  # type: ignore
        self, inputs: Union[tf.Tensor, TabularData], targets, training=False, **kwargs
    ) -> tf.Tensor:
        losses = []

        if isinstance(inputs, dict) and not all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            filtered_kwargs = filter_kwargs(
                dict(training=training), self, filter_positional_or_keyword=False
            )
            predictions = self(inputs, **filtered_kwargs)
        else:
            predictions = inputs

        for name, task in self.prediction_task_dict.items():
            loss = task.compute_loss(predictions, targets, training=training, **kwargs)
            losses.append(loss * self._task_weight_dict[name])

        return self.loss_reduction(losses)

    def metric_results(self, mode=None):
        def name_fn(x):
            return "_".join([mode, x]) if mode else x

        metrics = {
            name_fn(name): task.metric_results() for name, task in self.prediction_task_dict.items()
        }

        return _output_metrics(metrics)

    def metric_result_dict(self, mode=None):
        results = {}
        for name, task in self.prediction_task_dict.items():
            results.update(task.metric_results(mode=mode))

        return results

    def reset_metrics(self):
        for task in self.prediction_task_dict.values():
            task.reset_metrics()

    @property
    def task_blocks(self) -> Dict[str, Optional[Layer]]:
        return {name: task.task_block for name, task in self.prediction_task_dict.items()}

    @property
    def task_names(self) -> List[str]:
        return [name for name in self.prediction_task_dict]

    @property
    def metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        outputs = {}
        for name, task in self.parallel_dict.items():
            outputs.update({metric.name: metric for metric in task.metrics})

        return outputs

    def repr_ignore(self) -> List[str]:
        return ["prediction_tasks", "parallel_layers"]

    def _set_context(self, context: "BlockContext"):
        for task in self.prediction_task_dict.values():
            task._set_context(context)
        super(ParallelPredictionBlock, self)._set_context(context)

    @classmethod
    def from_config(cls, config, **kwargs):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["body", "prediction_tasks"])

        if "schema" in config:
            config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])

        config["loss_reduction"] = getattr(tf, config["loss_reduction"])

        prediction_tasks = config.pop("prediction_tasks", [])

        return cls(*prediction_tasks, **config)

    def get_config(self):
        config = super().get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self, config, ["body", "loss_reduction", "prediction_tasks"]
        )
        if self.task_weights:
            config["task_weights"] = self.task_weights

        return config
