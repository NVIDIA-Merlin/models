#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import collections
import inspect
import os
import sys
import warnings
from collections.abc import Sequence as SequenceCollection
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

import six
import tensorflow as tf
from keras.engine.compile_utils import MetricsContainer
from keras.utils.losses_utils import cast_losses_to_common_dtype
from packaging import version
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import unpack_x_y_sample_weight

# This is to handle TensorFlow 2.11/2.12 Saving V3 triggering with model pickle
try:
    from keras.saving.experimental import saving_lib  # 2.11
except ImportError:
    try:
        from keras.saving import saving_lib  # 2.12
    except ImportError:
        saving_lib = None

import merlin.io
from merlin.models.io import save_merlin_metadata
from merlin.models.tf.core.base import Block, ModelContext, NoOp, PredictionOutput, is_input_block
from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock
from merlin.models.tf.core.prediction import Prediction, PredictionContext, TensorLike
from merlin.models.tf.core.tabular import TabularBlock
from merlin.models.tf.distributed.backend import hvd, hvd_installed
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.loader import Loader
from merlin.models.tf.losses.base import loss_registry
from merlin.models.tf.metrics import metrics_registry
from merlin.models.tf.metrics.evaluation import MetricType
from merlin.models.tf.metrics.topk import TopKMetricsAggregator, filter_topk_metrics, split_metrics
from merlin.models.tf.models.utils import parse_prediction_blocks
from merlin.models.tf.outputs.base import ModelOutput, ModelOutputType
from merlin.models.tf.outputs.contrastive import ContrastiveOutput
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.transforms.features import PrepareFeatures, expected_input_cols_from_schema
from merlin.models.tf.transforms.sequence import SequenceTransform
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.search_utils import find_all_instances_in_layers
from merlin.models.tf.utils.tf_utils import (
    call_layer,
    get_sub_blocks,
    maybe_serialize_keras_objects,
)
from merlin.models.utils import schema_utils
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import ColumnSchema, Schema, Tags

if TYPE_CHECKING:
    from merlin.models.tf.core.encoder import Encoder
    from merlin.models.tf.core.index import TopKIndexBlock

METRICS_PARAMETERS_DOCSTRING = """
            The tasks metrics can be provided in different ways.
            If there is a single task, all metrics are assigned to that task.
            If there is more than one task, then we accept different ways to assign
            metrics for each task:
              1. If a single tf.keras.metrics.Metric or a list/tuple of Metric is provided,
                 the metrics are cloned for each task.
              2. If a list/tuple of list/tuple of Metric is provided and the number of nested
                 lists is the same as the number of tasks, it is assumed that each nested list
                 is associated to a task. By convention, Keras sorts tasks by name, so keep
                 that in mind when ordering your nested lists of metrics.
              3. If a dict of metrics is passed, it is expected that the keys match the name
                 of the tasks and values are Metric or list/tuple of Metric.
                 For example, if PredictionTask (V1) is being used, the task names should
                 be like "click/binary_classification_task", "rating/regression_task".
                 If OutputBlock (V2) is used, the task names should be like
                 "click/binary_output", "rating/regression_output"
"""

LOSS_PARAMETERS_DOCSTRINGS = """Can be either a single loss (str or tf.keras.losses.Loss)
            or a dict whose keys match the model tasks names.
            For example, if PredictionTask (V1) is being used, the task names should
            be like "click/binary_classification_task", "rating/regression_task".
            If OutputBlock (V2) is used, the task names should be like
            "click/binary_output", "rating/regression_output"
"""


class MetricsComputeCallback(tf.keras.callbacks.Callback):
    """Callback that handles when to compute metrics."""

    def __init__(self, train_metrics_steps=1, **kwargs):
        self.train_metrics_steps = train_metrics_steps
        self._is_fitting = False
        self._is_first_batch = True
        super().__init__(**kwargs)

    def on_train_begin(self, logs=None):
        self._is_fitting = True

    def on_train_end(self, logs=None):
        self._is_fitting = False

    def on_epoch_begin(self, epoch, logs=None):
        self._is_first_batch = True

    def on_train_batch_begin(self, batch, logs=None):
        value = self.train_metrics_steps > 0 and (
            self._is_first_batch or batch % self.train_metrics_steps == 0
        )
        self.model._should_compute_train_metrics_for_batch.assign(value)

    def on_train_batch_end(self, batch, logs=None):
        self._is_first_batch = False


def get_output_schema(export_path: str) -> Schema:
    """Compute Output Schema

    Parameters
    ----------
    export_path : str
        Path to saved model directory

    Returns
    -------
    Schema
        Output Schema representing model outputs
    """
    model = tf.keras.models.load_model(export_path)
    signature = model.signatures["serving_default"]

    output_schema = Schema()
    for output_name, output_spec in signature.structured_outputs.items():
        col_schema = ColumnSchema(output_name, dtype=output_spec.dtype.as_numpy_dtype)
        shape = output_spec.shape
        if shape.rank > 1 and (shape[1] is not None and shape[1] > 1):
            col_schema = ColumnSchema(
                output_name,
                dtype=output_spec.dtype.as_numpy_dtype,
                dims=(None, shape[1]),
            )

        output_schema.column_schemas[output_name] = col_schema

    return output_schema


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ModelBlock(Block, tf.keras.Model):
    """Block that extends `tf.keras.Model` to make it saveable."""

    def __init__(self, block: Block, prep_features: Optional[bool] = True, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        if hasattr(self, "set_schema"):
            block_schema = getattr(block, "schema", None)
            self.set_schema(block_schema)

        self.prep_features = prep_features
        self._prepare_features = PrepareFeatures(self.schema) if self.prep_features else NoOp()

    def call(self, inputs, **kwargs):
        inputs = self._prepare_features(inputs)
        if "features" not in kwargs:
            kwargs["features"] = inputs
        outputs = call_layer(self.block, inputs, **kwargs)
        return outputs

    def build(self, input_shapes):
        self._prepare_features.build(input_shapes)
        input_shapes = self._prepare_features.compute_output_shape(input_shapes)

        self.block.build(input_shapes)

        if not hasattr(self.build, "_is_default"):
            self._build_input_shape = input_shapes
        self.built = True

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        train_metrics_steps=1,
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, **kwargs)

        validation_data = _maybe_convert_merlin_dataset(
            validation_data, batch_size, shuffle=shuffle, **kwargs
        )
        callbacks = self._add_metrics_callback(callbacks, train_metrics_steps)
        fit_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "kwargs", "train_metrics_steps", "__class__"]
        }

        return super().fit(**fit_kwargs)

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, **kwargs)

        return super().evaluate(
            x,
            y,
            batch_size,
            verbose,
            sample_weight,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
            return_dict,
            **kwargs,
        )

    def compute_output_shape(self, input_shape):
        input_shape = self._prepare_features.compute_output_shape(input_shape)
        return self.block.compute_output_shape(input_shape)

    @property
    def schema(self) -> Schema:
        return self.block.schema

    @classmethod
    def from_config(cls, config, custom_objects=None):
        block = tf.keras.utils.deserialize_keras_object(config.pop("block"))

        return cls(block, **config)

    def get_config(self):
        return {"block": tf.keras.utils.serialize_keras_object(self.block)}

    def _set_save_spec(self, inputs, args=None, kwargs=None):
        # We need to overwrite this in order to fix a Keras-bug in TF<2.9
        super()._set_save_spec(inputs, args, kwargs)

        if version.parse(tf.__version__) < version.parse("2.9.0"):
            # Keras will interpret kwargs like `features` & `targets` as
            # required args, which is wrong. This is a workaround.
            _arg_spec = self._saved_model_arg_spec
            self._saved_model_arg_spec = ([_arg_spec[0][0]], _arg_spec[1])


class BaseModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)

        # Initializing model control flags controlled by MetricsComputeCallback()
        self._should_compute_train_metrics_for_batch = tf.Variable(
            dtype=tf.bool,
            name="should_compute_train_metrics_for_batch",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            initial_value=lambda: True,
        )

    def compile(
        self,
        optimizer="rmsprop",
        loss: Optional[Union[str, Loss, Dict[str, Union[str, Loss]]]] = None,
        metrics: Optional[
            Union[
                MetricType,
                Sequence[MetricType],
                Sequence[Sequence[MetricType]],
                Dict[str, MetricType],
                Dict[str, Sequence[MetricType]],
            ]
        ] = None,
        loss_weights=None,
        weighted_metrics: Optional[
            Union[
                MetricType,
                Sequence[MetricType],
                Sequence[Sequence[MetricType]],
                Dict[str, MetricType],
                Dict[str, Sequence[MetricType]],
            ]
        ] = None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        """Configures the model for training.
        Example:
        ```python
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.FalseNegatives()])
        ```
        Args:
            optimizer: String (name of optimizer) or optimizer instance. See
              `tf.keras.optimizers`.
            loss: Optional[Union[str, Loss, Dict[str, Union[str, Loss]]]] = None
              losses : Optional[Union[str, Loss, Dict[str, Union[str, Loss]]]], optional
              {LOSS_PARAMETERS_DOCSTRINGS}
              See `tf.keras.losses`. A loss
              function is any callable with the signature `loss = fn(y_true,
              y_pred)`, where `y_true` are the ground truth values, and
              `y_pred` are the model's predictions.
              `y_true` should have shape
              `(batch_size, d0, .. dN)` (except in the case of
              sparse loss functions such as
              sparse categorical crossentropy which expects integer arrays of shape
              `(batch_size, d0, .. dN-1)`).
              `y_pred` should have shape `(batch_size, d0, .. dN)`.
              The loss function should return a float tensor.
              If a custom `Loss` instance is
              used and reduction is set to `None`, return value has shape
              `(batch_size, d0, .. dN-1)` i.e. per-sample or per-timestep loss
              values; otherwise, it is a scalar. If the model has multiple outputs,
              you can use a different loss on each output by passing a dictionary
              or a list of losses. The loss value that will be minimized by the
              model will then be the sum of all individual losses, unless
              `loss_weights` is specified.
            loss_weights: Optional list or dictionary specifying scalar coefficients
              (Python floats) to weight the loss contributions of different model
              outputs. The loss value that will be minimized by the model will then
              be the *weighted sum* of all individual losses, weighted by the
              `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping to the model's
                  outputs (Keras sorts tasks by name).
                  If a dict, it is expected to map output names (strings)
                  to scalar coefficients.
            metrics: Optional[ Union[ MetricType, Sequence[MetricType],
                  Sequence[Sequence[MetricType]],
                  Dict[str, MetricType], Dict[str, Sequence[MetricType]], ] ], optional
                  {METRICS_PARAMETERS_DOCSTRING}
            weighted_metrics: Optional[ Union[ MetricType, Sequence[MetricType],
                  Sequence[Sequence[MetricType]],
                  Dict[str, MetricType], Dict[str, Sequence[MetricType]], ] ], optional
                  List of metrics to be evaluated and weighted by
                 `sample_weight` or `class_weight` during training and testing.
                 {METRICS_PARAMETERS_DOCSTRING}
            run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
              logic will not be wrapped in a `tf.function`. Recommended to leave
              this as `None` unless your `Model` cannot be run inside a
              `tf.function`. `run_eagerly=True` is not supported when using
              `tf.distribute.experimental.ParameterServerStrategy`.
            steps_per_execution: Int. Defaults to 1. The number of batches to run
              during each `tf.function` call. Running multiple batches inside a
              single `tf.function` call can greatly improve performance on TPUs or
              small models with a large Python overhead. At most, one full epoch
              will be run each execution. If a number larger than the size of the
              epoch is passed, the execution will be truncated to the size of the
              epoch. Note that if `steps_per_execution` is set to `N`,
              `Callback.on_batch_begin` and `Callback.on_batch_end` methods will
              only be called every `N` batches (i.e. before/after each `tf.function`
              execution).
            jit_compile: If `True`, compile the model training step with XLA.
              [XLA](https://www.tensorflow.org/xla) is an optimizing compiler for
              machine learning.
              `jit_compile` is not enabled for by default.
              This option cannot be enabled with `run_eagerly=True`.
              Note that `jit_compile=True` is
              may not necessarily work for all models.
              For more information on supported operations please refer to the
              [XLA documentation](https://www.tensorflow.org/xla).
              Also refer to
              [known XLA issues](https://www.tensorflow.org/xla/known_issues) for
              more details.
            **kwargs: Arguments supported for backwards compatibility only.
        """

        num_v1_blocks = len(self.prediction_tasks)
        num_v2_blocks = len(self.model_outputs)

        if num_v1_blocks > 1 and num_v2_blocks > 1:
            raise ValueError(
                "You cannot use both `prediction_tasks` and `prediction_blocks` at the same time.",
                "`prediction_tasks` is deprecated and will be removed in a future version.",
            )

        if num_v1_blocks > 0:
            self.output_names = [task.task_name for task in self.prediction_tasks]
        else:
            self.output_names = [block.full_name for block in self.model_outputs]

        # This flag will make Keras change the metric-names which is not needed in v2
        from_serialized = kwargs.pop("from_serialized", num_v2_blocks > 0)

        if hvd_installed and hvd.size() > 1:
            # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
            # uses hvd.DistributedOptimizer() to compute gradients.
            kwargs.update({"experimental_run_tf_function": False})

        super(BaseModel, self).compile(
            optimizer=self._create_optimizer(optimizer),
            loss=self._create_loss(loss),
            metrics=self._create_metrics(metrics, weighted=False),
            weighted_metrics=self._create_metrics(weighted_metrics, weighted=True),
            run_eagerly=run_eagerly,
            loss_weights=loss_weights,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            from_serialized=from_serialized,
            **kwargs,
        )

    def _create_optimizer(self, optimizer):
        def _create_single_distributed_optimizer(opt):
            opt_config = opt.get_config()

            if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_config = opt.learning_rate.get_config()
                lr_config["initial_learning_rate"] *= hvd.size()
                opt_config["lr"] = opt.learning_rate.__class__.from_config(lr_config)
            else:
                opt_config["lr"] = opt.learning_rate * hvd.size()

            opt = opt.__class__.from_config(opt_config)

            return hvd.DistributedOptimizer(opt)

        if version.parse(tf.__version__) < version.parse("2.11.0"):
            optimizer = tf.keras.optimizers.get(optimizer)
        else:
            optimizer = tf.keras.optimizers.get(optimizer, use_legacy_optimizer=True)

        if hvd_installed and hvd.size() > 1:
            if optimizer.__module__.startswith("horovod"):
                # do nothing if the optimizer is already wrapped in hvd.DistributedOptimizer
                pass
            elif isinstance(optimizer, merlin.models.tf.MultiOptimizer):
                for pair in (
                    optimizer.optimizers_and_blocks + optimizer.update_optimizers_and_blocks
                ):
                    pair.optimizer = _create_single_distributed_optimizer(pair.optimizer)
            else:
                optimizer = _create_single_distributed_optimizer(optimizer)

        return optimizer

    def _create_metrics(
        self,
        metrics: Optional[
            Union[
                MetricType,
                Sequence[MetricType],
                Sequence[Sequence[MetricType]],
                Dict[str, MetricType],
                Dict[str, Sequence[MetricType]],
            ]
        ] = None,
        weighted: bool = False,
    ) -> Union[MetricType, Dict[str, Sequence[MetricType]]]:
        """Creates metrics for the model tasks (defined by using either
           PredictionTask (V1) or OutputBlock (V2)).

        Parameters
        ----------
        metrics : {METRICS_PARAMETERS_DOCSTRING}
        weighted : bool, optional
            Whether these are the metrics or weighted_metrics, by default False (metrics)

        Returns
        -------
        Union[MetricType, Dict[str, Sequence[MetricType]]]
            Returns the metrics organized by task
        """
        out = {}

        def parse_str_metrics(metrics):
            if isinstance(metrics, str):
                metrics = metrics_registry.parse(metrics)
            elif isinstance(metrics, (tuple, list)):
                metrics = list([parse_str_metrics(m) for m in metrics])
            elif isinstance(metrics, dict):
                metrics = {k: parse_str_metrics(v) for k, v in metrics.items()}
            return metrics

        num_v1_blocks = len(self.prediction_tasks)
        if isinstance(metrics, dict):
            out = metrics
            out = {
                k: parse_str_metrics([(v)] if isinstance(v, (str, Metric)) else v)
                for k, v in out.items()
            }

        elif isinstance(metrics, (list, tuple)):
            # Retrieve top-k metrics & wrap them in TopKMetricsAggregator
            topk_metrics, topk_aggregators, other_metrics = split_metrics(metrics)
            metrics = other_metrics + topk_aggregators
            if len(topk_metrics) > 0:
                if len(topk_metrics) == 1:
                    metrics.append(topk_metrics[0])
                else:
                    metrics.append(TopKMetricsAggregator(*topk_metrics))

            def task_metrics(metrics, tasks):
                out_task_metrics = {}
                for i, task in enumerate(tasks):
                    if any([isinstance(m, (tuple, list)) for m in metrics]):
                        if len(metrics) == len(tasks):
                            task_metrics = metrics[i]
                            task_metrics = parse_str_metrics(task_metrics)
                            if isinstance(task_metrics, (str, Metric)):
                                task_metrics = [task_metrics]
                        else:
                            raise ValueError(
                                "If metrics are lists of lists, the number of"
                                "sub-lists must match number of tasks."
                            )
                    else:
                        task_metrics = list(parse_str_metrics(m) for m in metrics)
                        # Cloning metrics for each task
                        task_metrics = list([m.from_config(m.get_config()) for m in task_metrics])

                    task_name = (
                        task.full_name if isinstance(tasks[0], ModelOutput) else task.task_name
                    )
                    out_task_metrics[task_name] = task_metrics
                return out_task_metrics

            if num_v1_blocks > 0:
                if num_v1_blocks == 1:
                    out[self.prediction_tasks[0].task_name] = parse_str_metrics(metrics)
                else:
                    out = task_metrics(metrics, self.prediction_tasks)
            else:
                if len(self.model_outputs) == 1:
                    out[self.model_outputs[0].full_name] = parse_str_metrics(metrics)
                else:
                    out = task_metrics(metrics, self.model_outputs)

        elif isinstance(metrics, (str, Metric)):
            metrics = parse_str_metrics(metrics)
            if num_v1_blocks == 0:
                for prediction_name, prediction_block in self.outputs_by_name().items():
                    # Cloning the metric for every task
                    out[prediction_name] = [metrics.from_config(metrics.get_config())]
            else:
                out = metrics

        elif metrics is None:
            if not weighted:
                # Get default metrics
                for task_name, task in self.prediction_tasks_by_name().items():
                    out[task_name] = [
                        m()
                        if inspect.isclass(m) or type(task.DEFAULT_METRICS[0]) == partial
                        else parse_str_metrics(m)
                        for m in task.DEFAULT_METRICS
                    ]

                for prediction_name, prediction_block in self.outputs_by_name().items():
                    out[prediction_name] = parse_str_metrics(prediction_block.default_metrics_fn())

        else:
            raise ValueError("Invalid metrics value.")

        if out:
            if num_v1_blocks == 0:  # V2
                for prediction_name, prediction_block in self.outputs_by_name().items():
                    for metric in out[prediction_name]:
                        if len(self.model_outputs) > 1:
                            # Setting hierarchical metric names (column/task/metric_name)
                            metric._name = "/".join(
                                [
                                    prediction_block.full_name,
                                    f"weighted_{metric._name}" if weighted else metric._name,
                                ]
                            )
                        else:
                            if weighted:
                                metric._name = f"weighted_{metric._name}"

            for metric in tf.nest.flatten(out):
                # We ensure metrics passed to `compile()` are reset
                if metric:
                    metric.reset_state()
        else:
            out = None
        return out

    def _create_loss(
        self, losses: Optional[Union[str, Loss, Dict[str, Union[str, Loss]]]] = None
    ) -> Dict[str, Loss]:
        """Creates the losses for model tasks (defined by using either
           PredictionTask (V1) or OutputBlock (V2)).

        Parameters
        ----------
        losses : Optional[Union[str, Loss, Dict[str, Union[str, Loss]]]], optional
            {LOSS_PARAMETERS_DOCSTRINGS}

        Returns
        -------
        Dict[str, Loss]
            Returns a dict with the losses per task
        """
        out = {}

        if isinstance(losses, dict):
            out = losses

        elif isinstance(losses, (Loss, str)):
            if len(self.prediction_tasks) == 1:
                out = {task.task_name: losses for task in self.prediction_tasks}
            elif len(self.model_outputs) == 1:
                out = {task.name: losses for task in self.model_outputs}

        # If loss is not provided, use the defaults from the prediction-tasks.
        elif not losses:
            for task_name, task in self.prediction_tasks_by_name().items():
                out[task_name] = task.DEFAULT_LOSS

            for task_name, task in self.outputs_by_name().items():
                out[task_name] = task.default_loss

        for key in out:
            if isinstance(out[key], str) and out[key] in loss_registry:
                out[key] = loss_registry.parse(out[key])

        return out

    @property
    def prediction_tasks(self) -> List[PredictionTask]:
        from merlin.models.tf.prediction_tasks.base import PredictionTask

        results = find_all_instances_in_layers(self, PredictionTask)
        # Ensures tasks are sorted by name, so that they match the metrics
        # which are sorted the same way by Keras
        results = list(sorted(results, key=lambda x: x.task_name))

        return results

    def prediction_tasks_by_name(self) -> Dict[str, PredictionTask]:
        return {task.task_name: task for task in self.prediction_tasks}

    def prediction_tasks_by_target(self) -> Dict[str, List[PredictionTask]]:
        """Method to index the model's prediction tasks by target names.

        Returns
        -------
        Dict[str, List[PredictionTask]]
            List of prediction tasks.
        """
        outputs: Dict[str, Union[PredictionTask, List[PredictionTask]]] = {}
        for task in self.prediction_tasks:
            if task.target_name in outputs:
                if isinstance(outputs[task.target_name], list):
                    outputs[task.target].append(task)
                else:
                    outputs[task.target_name] = [outputs[task.target_name], task]
            outputs[task.target] = task

        return outputs

    @property
    def model_outputs(self) -> List[ModelOutput]:
        results = find_all_instances_in_layers(self, ModelOutput)
        # Ensures tasks are sorted by name, so that they match the metrics
        # which are sorted the same way by Keras
        results = list(sorted(results, key=lambda x: x.full_name))

        return results

    def outputs_by_name(self) -> Dict[str, ModelOutput]:
        return {task.full_name: task for task in self.model_outputs}

    def outputs_by_target(self) -> Dict[str, List[ModelOutput]]:
        """Method to index the model's prediction blocks by target names.

        Returns
        -------
        Dict[str, List[PredictionBlock]]
            List of prediction blocks.
        """
        outputs: Dict[str, List[ModelOutput]] = {}
        for task in self.model_outputs:
            if task.target in outputs:
                if isinstance(outputs[task.target], list):
                    outputs[task.target].append(task)
                else:
                    outputs[task.target] = [outputs[task.target], task]
            outputs[task.target] = task

        return outputs

    def call_train_test(
        self,
        x: TabularData,
        y: Optional[Union[tf.tensor, TabularData]] = None,
        sample_weight=Optional[Union[float, tf.Tensor]],
        training: bool = False,
        testing: bool = False,
        **kwargs,
    ) -> Union[Prediction, PredictionOutput]:
        """Apply the model's call method during Train or Test modes and prepare
        Prediction (v2) or PredictionOutput (v1 -
        depreciated) objects

        Parameters
        ----------
        x : TabularData
            Dictionary of raw input features.
        y : Union[tf.tensor, TabularData], optional
            Target tensors, by default None
        training : bool, optional
            Flag for train mode, by default False
        sample_weight : Union[float, tf.Tensor], optional
            Sample weights to be used by the loss and by weighted_metrics
        testing : bool, optional
            Flag for test mode, by default False

        Returns
        -------
        Union[Prediction, PredictionOutput]
        """

        forward = self(
            x,
            targets=y,
            training=training,
            testing=testing,
            **kwargs,
        )
        if not (self.prediction_tasks or self.model_outputs):
            return PredictionOutput(forward, y)

        predictions, targets, sample_weights, output = {}, {}, {}, None
        # V1
        if self.prediction_tasks:
            for task in self.prediction_tasks:
                task_x = forward
                if isinstance(forward, dict) and task.task_name in forward:
                    task_x = forward[task.task_name]

                if isinstance(task_x, PredictionOutput):
                    output = task_x
                    task_y = output.targets
                    task_x = output.predictions
                    task_sample_weight = (
                        sample_weight if output.sample_weight is None else output.sample_weight
                    )
                else:
                    task_y = y[task.target_name] if isinstance(y, dict) and y else y
                    task_sample_weight = sample_weight

                targets[task.task_name] = task_y
                predictions[task.task_name] = task_x
                sample_weights[task.task_name] = task_sample_weight

            self.adjust_predictions_and_targets(predictions, targets, sample_weights)

            if len(predictions) == 1 and len(targets) == 1:
                predictions = list(predictions.values())[0]
                targets = list(targets.values())[0]
                sample_weights = list(sample_weights.values())[0]

            if output:
                return output.copy_with_updates(predictions, targets, sample_weight=sample_weights)
            else:
                return PredictionOutput(predictions, targets, sample_weight=sample_weights)

        # V2
        for task in self.model_outputs:
            task_x = forward
            if isinstance(forward, dict) and task.full_name in forward:
                task_x = forward[task.full_name]
            if isinstance(task_x, Prediction):
                output = task_x
                task_y = output.targets
                task_x = output.outputs
                task_sample_weight = (
                    sample_weight if output.sample_weight is None else output.sample_weight
                )
            else:
                task_y = y[task.target] if isinstance(y, dict) and y else y
                task_sample_weight = sample_weight

            targets[task.full_name] = task_y
            predictions[task.full_name] = task_x
            sample_weights[task.full_name] = task_sample_weight

        self.adjust_predictions_and_targets(predictions, targets, sample_weights)

        return Prediction(predictions, targets, sample_weights)

    def _extract_masked_predictions(self, prediction: TensorLike):
        """Extracts the prediction scores corresponding to masked positions (targets).

        This method assumes that the input predictions tensor is 3-D and contains a mask
        indicating the positions of the targets. It requires that the mask information has
        exactly one masked position per input sequence. The method returns a 2-D dense tensor
        containing the prediction score corresponding to each masked position.

        Parameters
        ----------
        prediction : TensorLike
            A 3-D dense tensor of predictions, with a mask indicating the positions of the targets.

        Returns
        -------
        tf.Tensor
            A 2-D dense tensor of prediction scores, with one score per input.

        Raises
        ------
        ValueError
            If the mask does not have exactly one masked position per input sequence.
        """
        num_preds_per_example = tf.reduce_sum(tf.cast(prediction._keras_mask, tf.int32), axis=-1)
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(
                    num_preds_per_example,
                    1,
                    message="If targets are scalars (1-D) and predictions are"
                    " sequential (3-D), the predictions mask should contain"
                    " one masked position per example",
                )
            ]
        ):
            return tf.boolean_mask(prediction, prediction._keras_mask)

    def _adjust_dense_predictions_and_targets(
        self,
        prediction: tf.Tensor,
        target: TensorLike,
        sample_weight: TensorLike,
    ):
        """Adjusts the dense predictions tensor, the target tensor and sample_weight tensor
        to ensure compatibility with most Keras losses and metrics.

        This method applies the following transformations to the target and prediction tensors:
        - Converts ragged targets and their masks to dense format.
        - Copies the target mask to the prediction mask, if defined.
        - If predictions are sequential (3-D) and targets are scalar (1-D), extracts the predictions
        at target positions using the predictions mask.
        - One-hot encodes targets if their rank is one less than the rank of predictions.
        - Ensures that targets have the same shape and dtype as predictions.

        Parameters
        ----------
        prediction : tf.Tensor
            The prediction tensor as a dense tensor.
        target : TensorLike
            The target tensor that can be either a dense or ragged tensor.
        sample_weight : TensorLike
            The sample weight tensor that can be either a dense or ragged tensor.

        Returns:
        --------
            A tuple of the adjusted prediction, target, and sample_weight tensors,
            with the same dtype and shape.
        """
        if isinstance(target, tf.RaggedTensor):
            # Converts ragged targets (and ragged mask) to dense
            dense_target_mask = None
            if getattr(target, "_keras_mask", None) is not None:
                dense_target_mask = target._keras_mask.to_tensor()
            target = target.to_tensor()
            if dense_target_mask is not None:
                target._keras_mask = dense_target_mask

        if isinstance(sample_weight, tf.RaggedTensor):
            sample_weight = sample_weight.to_tensor()

        if prediction.shape.ndims == 2:
            # Removes the mask information as the sequence is summarized into
            # a single vector.
            prediction._keras_mask = None

        elif getattr(target, "_keras_mask", None) is not None:
            # Copies the mask from the targets to the predictions
            # because Keras considers the prediction mask in loss
            # and metrics computation
            if isinstance(target._keras_mask, tf.RaggedTensor):
                target._keras_mask = target._keras_mask.to_tensor()
            prediction._keras_mask = target._keras_mask

        # Ensuring targets and preds have the same dtype
        target = tf.cast(target, prediction.dtype)

        # If targets are scalars (1-D) and predictions are sequential (3-D),
        # extract predictions at target position because Keras expects
        # predictions and targets to have the same shape.
        if getattr(prediction, "_keras_mask", None) is not None:
            rank_check = tf.logical_and(
                tf.logical_and(tf.rank(target) > 0, tf.shape(target)[-1] == 1),
                tf.equal(tf.rank(prediction), 3),
            )
            prediction = tf.cond(
                rank_check, lambda: self._extract_masked_predictions(prediction), lambda: prediction
            )

        # Ensuring targets are one-hot encoded if they are not
        condition = tf.logical_and(
            tf.logical_and(tf.rank(target) > 0, tf.shape(target)[-1] == 1),
            tf.shape(prediction)[-1] > 1,
        )
        target = tf.cond(
            condition,
            lambda: tf.one_hot(
                tf.cast(target, tf.int32),
                tf.shape(prediction)[-1],
                dtype=prediction.dtype,
            ),
            lambda: target,
        )
        # Makes target shape equal to the predictions tensor, as shape is lost after tf.cond
        target = tf.reshape(target, tf.shape(prediction))

        return prediction, target, sample_weight

    def _adjust_ragged_predictions_and_targets(
        self,
        prediction: tf.RaggedTensor,
        target: TensorLike,
        sample_weight: TensorLike,
    ):
        """Adjusts the prediction (ragged tensor), target and sample weight
        to ensure compatibility with most Keras losses and metrics.

        This methods applies the following transformations to the target and prediction tensors:
        - Select ragged targets based on the mask information, if defined.
        - Remove mask information from the ragged targets and predictions.
        - One-hot encode targets if their rank is one less than the rank of predictions.
        - Ensure that targets have the same shape and dtype as predictions.

        Parameters
        ----------
        prediction : tf.RaggedTensor
            The prediction tensor as a ragged tensor.
        target : TensorLike
            The target tensor that can be either a dense or ragged tensor.
        sample_weight : TensorLike
            The sample weight tensor that can be either a dense or ragged tensor.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing the adjusted prediction, target and sample_weight tensors.
        """
        target_mask = None
        if getattr(target, "_keras_mask", None) is not None:
            target_mask = target._keras_mask

        if isinstance(target, tf.RaggedTensor) and target_mask is not None:
            # Select targets at masked positions and return
            # a ragged tensor.
            target = tf.ragged.boolean_mask(
                target, target_mask.with_row_splits_dtype(target.row_splits.dtype)
            )

        # Ensuring targets and preds have the same dtype
        target = tf.cast(target, prediction.dtype)

        # Align sample_weight with the ragged target tensor
        if isinstance(target, tf.RaggedTensor) and sample_weight is not None:
            if isinstance(sample_weight, tf.RaggedTensor):
                # sample_weight is a 2-D tensor, weights in the same sequence are different
                if target_mask is not None:
                    # Select sample weights at masked positions and return a ragged tensor.
                    sample_weight = tf.ragged.boolean_mask(
                        sample_weight,
                        target_mask.with_row_splits_dtype(sample_weight.row_splits.dtype),
                    )
            else:
                # sample_weight is a 1-D tensor, one weight value per sequence
                # repeat the weight value for each masked target position
                row_lengths = tf.constant(target.row_lengths(), dtype=tf.int64)
                sample_weight = tf.repeat(sample_weight, row_lengths)

        # Take the flat values of predictions, targets and sample weihts as Keras
        # losses does not support RaggedVariantTensor on GPU:
        prediction = prediction.flat_values
        if isinstance(target, tf.RaggedTensor):
            target = target.flat_values
        if isinstance(sample_weight, tf.RaggedTensor):
            sample_weight = sample_weight.flat_values

        # Ensuring targets are one-hot encoded if they are not
        condition = tf.logical_and(
            tf.logical_and(tf.rank(target) > 0, tf.shape(target)[-1] == 1),
            tf.shape(prediction)[-1] > 1,
        )
        target = tf.cond(
            condition,
            lambda: tf.one_hot(
                tf.cast(target, tf.int32),
                tf.shape(prediction)[-1],
                dtype=prediction.dtype,
            ),
            lambda: target,
        )

        # Makes target shape equal to the predictions tensor, as shape is lost after tf.cond
        target = tf.reshape(target, tf.shape(prediction))

        return prediction, target, sample_weight

    def adjust_predictions_and_targets(
        self,
        predictions: Dict[str, TensorLike],
        targets: Optional[Union[TensorLike, Dict[str, TensorLike]]],
        sample_weights: Optional[Union[TensorLike, Dict[str, TensorLike]]],
    ):
        """Adjusts the predictions and targets to ensure compatibility with
        most Keras losses and metrics.

        If the predictions are ragged tensors, `_adjust_ragged_predictions_and_targets` is called,
        otherwise `_adjust_dense_predictions_and_targets` is called.

        Parameters
        ----------
        predictions : Dict[str, TensorLike]
            A dictionary with predictions for the tasks.
        targets : Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]]
            A dictionary with targets for the tasks, or None if targets are not provided.
        sample_weights : Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]]
            A dictionary with sample weights for the tasks,
            or None if sample_weights are not provided.

        """
        if targets is None:
            return

        for k in targets:
            if isinstance(predictions[k], tf.RaggedTensor):
                (
                    predictions[k],
                    targets[k],
                    sample_weights[k],
                ) = self._adjust_ragged_predictions_and_targets(
                    predictions[k], targets[k], sample_weights[k]
                )
            else:
                (
                    predictions[k],
                    targets[k],
                    sample_weights[k],
                ) = self._adjust_dense_predictions_and_targets(
                    predictions[k], targets[k], sample_weights[k]
                )

    def train_step(self, data):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            x, y, sample_weight = unpack_x_y_sample_weight(data)

            # Ensure that we don't have any ragged or sparse tensors passed at training time.
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], (tf.RaggedTensor, tf.SparseTensor)):
                        raise ValueError(
                            "Training with RaggedTensor or SparseTensor input features is "
                            "not supported. Please update your dataloader to pass a tuple "
                            "of dense tensors instead, (corresponding to the values and "
                            "row lengths of the ragged input feature). This will ensure that "
                            "the model can be saved with the correct input signature, "
                            "and served correctly. "
                            "This is because when ragged or sparse tensors are fed as inputs "
                            "the input feature names are currently lost in the saved model "
                            "input signature."
                        )

            if getattr(self, "train_pre", None):
                out = call_layer(self.train_pre, x, targets=y, features=x, training=True)
                if isinstance(out, Prediction):
                    x, y = out.outputs, out.targets
                elif isinstance(out, tuple):
                    assert (
                        len(out) == 2
                    ), "output of `pre` must be a 2-tuple of x, y or `Prediction` tuple"
                    x, y = out
                else:
                    x = out

            outputs = self.call_train_test(x, y, sample_weight=sample_weight, training=True)
            loss = self.compute_loss(x, outputs.targets, outputs.predictions, outputs.sample_weight)

        self._validate_target_and_loss(outputs.targets, loss)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        outputs = outputs.copy_with_updates(
            sample_weight=self._extract_positive_sample_weights(outputs.sample_weight)
        )
        metrics = self.train_compute_metrics(outputs, self.compiled_metrics)

        # Batch regularization loss
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))
        # Batch loss (the default loss metric from Keras is the incremental average per epoch,
        # not the actual batch loss)
        metrics["loss_batch"] = loss

        return metrics

    def test_step(self, data):
        """Custom test step using the `compute_loss` method."""

        x, y, sample_weight = unpack_x_y_sample_weight(data)

        if getattr(self, "test_pre", None):
            out = call_layer(self.test_pre, x, targets=y, features=x, testing=True)
            if isinstance(out, Prediction):
                x, y = out.outputs, out.targets
            elif isinstance(out, tuple):
                assert (
                    len(out) == 2
                ), "output of `pre` must be a 2-tuple of x, y or `Prediction` tuple"
                x, y = out
            else:
                x = out

        outputs = self.call_train_test(x, y, sample_weight=sample_weight, testing=True)

        if getattr(self, "pre_eval_topk", None) is not None:
            # During eval, the retrieval-task only returns positive scores
            # so we need to retrieve top-k negative scores to compute the loss
            outputs = self.pre_eval_topk.call_outputs(outputs)

        loss = self.compute_loss(x, outputs.targets, outputs.predictions, outputs.sample_weight)

        outputs = outputs.copy_with_updates(
            sample_weight=self._extract_positive_sample_weights(outputs.sample_weight)
        )
        metrics = self.compute_metrics(outputs)

        # Batch regularization loss
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))
        # Batch loss (the default loss metric from Keras is the incremental average per epoch,
        # not the actual batch loss)
        metrics["loss_batch"] = loss

        return metrics

    def predict_step(self, data):
        x, _, _ = unpack_x_y_sample_weight(data)

        if getattr(self, "predict_pre", None):
            out = call_layer(self.predict_pre, x, features=x, training=False)
            if isinstance(out, Prediction):
                x = out.outputs
            elif isinstance(out, tuple):
                assert (
                    len(out) == 2
                ), "output of `pre` must be a 2-tuple of x, y or `Prediction` tuple"
                x, y = out
            else:
                x = out

        return self(x, training=False)

    def train_compute_metrics(self, outputs: PredictionOutput, compiled_metrics: MetricsContainer):
        """Returns metrics for the outputs of this step.

        Re-computing metrics every `train_metrics_steps` steps.
        """
        # Compiled_metrics as an argument here because it is re-defined by `model.compile()`
        # And checking `self.compiled_metrics` inside this function results in a reference to
        # a deleted version of `compiled_metrics` if the model is re-compiled.
        return tf.cond(
            self._should_compute_train_metrics_for_batch,
            lambda: self.compute_metrics(outputs, compiled_metrics),
            lambda: self.metrics_results(),
        )

    def compute_metrics(
        self,
        prediction_outputs: PredictionOutput,
        compiled_metrics: Optional[MetricsContainer] = None,
    ) -> Dict[str, tf.Tensor]:
        """Overrides Model.compute_metrics() for some custom behaviour
           like compute metrics each N steps during training
           and allowing to feed additional information required by specific metrics

        Parameters
        ----------
        prediction_outputs : PredictionOutput
            Contains properties with targets and predictions
        compiled_metrics : MetricsContainer
            The metrics container to compute metrics on.
            If not provided, uses self.compiled_metrics

        Returns
        -------
        Dict[str, tf.Tensor]
            Dict with the metrics values
        """
        if compiled_metrics is None:
            compiled_metrics = self.compiled_metrics

        # This ensures that compiled metrics are built
        # to make self.compiled_metrics.metrics available
        if not compiled_metrics.built:
            compiled_metrics.build(prediction_outputs.predictions, prediction_outputs.targets)

        # Providing label_relevant_counts for TopkMetrics, as metric.update_state()
        # should have standard signature for better compatibility with Keras methods
        # like self.compiled_metrics.update_state()
        if hasattr(prediction_outputs, "label_relevant_counts"):
            for topk_metric in filter_topk_metrics(compiled_metrics.metrics):
                topk_metric.label_relevant_counts = prediction_outputs.label_relevant_counts

        compiled_metrics.update_state(
            prediction_outputs.targets,
            prediction_outputs.predictions,
            prediction_outputs.sample_weight,
        )

        # Returns the current value of metrics
        metrics = self.metrics_results()
        return metrics

    def metrics_results(self) -> Dict[str, tf.Tensor]:
        """Logic to consolidate metrics results
        extracted from standard Keras Model.compute_metrics()

        Returns
        -------
        Dict[str, tf.Tensor]
            Dict with the metrics values
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    @property
    def input_schema(self) -> Optional[Schema]:
        """Get the input schema if it's defined.

        Returns
        -------
        Optional[Schema]
            Schema corresponding to the inputs of the model
        """
        schema = getattr(self, "schema", None)
        if isinstance(schema, Schema) and schema.column_names:
            target_tags = [Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION]
            return schema.excluding_by_tag(target_tags)

    def _maybe_set_schema(self, maybe_loader):
        """Try to set the correct schema on the model or loader.

        Parameters
        ----------
        maybe_loader : Union[Loader, Any]
            A Loader object or other valid input data to the model.

        Raises
        ------
        ValueError
            If the dataloader features do not match the model inputs
            and we're unable to automatically configure the dataloader
            to return only the required features
        """
        if isinstance(maybe_loader, Loader):
            loader = maybe_loader
            target_tags = [Tags.TARGET, Tags.BINARY_CLASSIFICATION, Tags.REGRESSION]
            if self.input_schema:
                loader_output_features = set(
                    loader.output_schema.excluding_by_tag(target_tags).column_names
                )
                model_input_features = set(self.input_schema.column_names)
                schemas_match = loader_output_features == model_input_features
                loader_is_superset = loader_output_features.issuperset(model_input_features)
                if not schemas_match and loader_is_superset and not loader.transforms:
                    # To ensure that the model receives only the features it requires.
                    loader.input_schema = self.input_schema + loader.input_schema.select_by_tag(
                        target_tags
                    )
            else:
                # Bind input schema from dataset to model,
                # to handle the case where this hasn't been set on an input block
                self.schema = loader.output_schema.excluding_by_tag(target_tags)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        train_metrics_steps=1,
        pre=None,
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, **kwargs)
        self._maybe_set_schema(x)

        if hasattr(x, "batch_size"):
            self._batch_size = x.batch_size

        validation_data = _maybe_convert_merlin_dataset(
            validation_data, batch_size, shuffle=shuffle, **kwargs
        )

        callbacks = self._add_metrics_callback(callbacks, train_metrics_steps)
        if hvd_installed and hvd.size() > 1:
            callbacks = self._add_horovod_callbacks(callbacks)

        # Horovod: if it's not worker 0, turn off logging.
        if hvd_installed and hvd.rank() != 0:
            verbose = 0  # noqa: F841

        fit_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "kwargs", "train_metrics_steps", "pre"] and not k.startswith("__")
        }

        if pre:
            self._reset_compile_cache()
            self.train_pre = pre
            if isinstance(self.train_pre, SequenceTransform):
                self.train_pre.configure_for_train()

        out = super().fit(**fit_kwargs)

        if pre:
            del self.train_pre

        return out

    def _validate_callbacks(self, callbacks):
        if callbacks is None:
            callbacks = []

        if isinstance(callbacks, SequenceCollection):
            callbacks = list(callbacks)
        else:
            callbacks = [callbacks]

        return callbacks

    def _add_metrics_callback(self, callbacks, train_metrics_steps):
        callbacks = self._validate_callbacks(callbacks)

        callback_types = [type(callback) for callback in callbacks]
        if MetricsComputeCallback not in callback_types:
            # Adding a callback to control metrics computation
            callbacks.append(MetricsComputeCallback(train_metrics_steps))

        return callbacks

    def _extract_positive_sample_weights(self, sample_weights):
        # 2-D sample weights are set for retrieval models to differentiate
        # between positive and negative candidates of the same sample.
        # For metrics calculation, we extract the sample weights of
        # the positive class (i.e. the first column)
        if sample_weights is None:
            return sample_weights

        if isinstance(sample_weights, tf.Tensor) and (len(sample_weights.shape) == 2):
            return tf.expand_dims(sample_weights[:, 0], -1)

        for name, weights in sample_weights.items():
            if isinstance(weights, dict):
                sample_weights[name] = self._extract_positive_sample_weights(weights)
            elif (weights is not None) and (len(weights.shape) == 2):
                sample_weights[name] = tf.expand_dims(weights[:, 0], -1)
        return sample_weights

    def _add_horovod_callbacks(self, callbacks):
        if not (hvd_installed and hvd.size() > 1):
            return callbacks

        callbacks = self._validate_callbacks(callbacks)

        callback_types = [type(callback) for callback in callbacks]
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        if hvd.callbacks.BroadcastGlobalVariablesCallback not in callback_types:
            callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        # Horovod: average metrics among workers at the end of every epoch.
        if hvd.callbacks.MetricAverageCallback not in callback_types:
            callbacks.append(hvd.callbacks.MetricAverageCallback())

        return callbacks

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        pre=None,
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, shuffle=False, **kwargs)

        if pre:
            self._reset_compile_cache()
            self.test_pre = pre
            if isinstance(self.test_pre, SequenceTransform):
                self.test_pre.configure_for_test()

        out = super().evaluate(
            x,
            y,
            batch_size,
            verbose,
            sample_weight,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
            return_dict,
            **kwargs,
        )

        if pre:
            del self.test_pre

        return out

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        pre=None,
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, shuffle=False, **kwargs)

        if pre:
            self._reset_compile_cache()
            self.predict_pre = pre

        out = super(BaseModel, self).predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        if pre:
            del self.predict_pre

        return out

    def batch_predict(
        self, dataset: merlin.io.Dataset, batch_size: int, **kwargs
    ) -> merlin.io.Dataset:
        """Batched prediction using the Dask.
        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to predict on.
        batch_size: int
            Batch size to use for prediction.
        Returns merlin.io.Dataset
        -------
        """
        if hasattr(dataset, "schema"):
            if not set(self.schema.column_names).issubset(set(dataset.schema.column_names)):
                raise ValueError(
                    f"Model schema {self.schema.column_names} does not match dataset schema"
                    + f" {dataset.schema.column_names}"
                )

        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        from merlin.models.tf.utils.batch_utils import TFModelEncode

        model_encode = TFModelEncode(self, batch_size=batch_size, **kwargs)
        predictions = dataset.map_partitions(model_encode)

        return merlin.io.Dataset(predictions)

    def save(self, *args, **kwargs):
        if hvd_installed and hvd.rank() != 0:
            return
        super().save(*args, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Model(BaseModel):
    """Merlin Model class

    Parameters
    ----------
    context : Optional[ModelContext], optional
        ModelContext is used to store/retrieve public variables across blocks,
        by default None.
    pre : Optional[BlockType], optional
        Optional `Block` instance to apply before the `call` method of the Two-Tower block
    post : Optional[BlockType], optional
        Optional `Block` instance to apply on both outputs of Two-tower model
        to output a single Tensor.
    schema : Optional[Schema], optional
        The `Schema` object with the input features.
    prep_features: Optional[bool]
        Whether this block should prepare list and scalar features
        from the dataloader format. By default True.
    """

    def __init__(
        self,
        *blocks: Block,
        context: Optional[ModelContext] = None,
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
        schema: Optional[Schema] = None,
        prep_features: Optional[bool] = True,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)

        context = context or ModelContext()
        if len(blocks) == 1 and isinstance(blocks[0], SequentialBlock):
            blocks = blocks[0].layers

        self.blocks = blocks
        for block in self.submodules:
            if hasattr(block, "_set_context"):
                block._set_context(context)

        self.pre = pre
        self.post = post
        self.context = context
        self._is_fitting = False
        self._batch_size = None

        if schema is not None:
            self.schema = schema
        else:
            input_block_schemas = [
                block.schema for block in self.submodules if getattr(block, "is_input", False)
            ]
            self.schema = sum(input_block_schemas, Schema())

        self.prep_features = prep_features

        self._prepare_features = PrepareFeatures(self.schema)
        self._frozen_blocks = set()

    def save(
        self,
        export_path: Union[str, os.PathLike],
        include_optimizer=True,
        save_traces=True,
    ) -> None:
        """Saves the model to export_path as a Tensorflow Saved Model.
        Along with merlin model metadata.

        Parameters
        ----------
        export_path : Union[str, os.PathLike]
            Path where model will be saved to
        include_optimizer : bool, optional
            If False, do not save the optimizer state, by default True
        save_traces : bool, optional
            When enabled, will store the function traces for each layer. This
            can be disabled, so that only the configs of each layer are
            stored, by default True
        """
        if hvd_installed and hvd.rank() != 0:
            return
        super().save(
            export_path,
            include_optimizer=include_optimizer,
            save_traces=save_traces,
            save_format="tf",
        )
        input_schema = self.schema
        output_schema = get_output_schema(export_path)
        save_merlin_metadata(export_path, input_schema, output_schema)

    @classmethod
    def load(cls, export_path: Union[str, os.PathLike]) -> "Model":
        """Loads a model that was saved with `model.save()`.

        Parameters
        ----------
        export_path : Union[str, os.PathLike]
            The path to the saved model.
        """
        return tf.keras.models.load_model(export_path)

    def _check_schema_and_inputs_matching(self, inputs):
        if isinstance(self.input_schema, Schema):
            model_expected_features = set(
                expected_input_cols_from_schema(self.input_schema, inputs)
            )
            call_input_features = set(inputs.keys())
            if model_expected_features != call_input_features:
                raise ValueError(
                    "Model called with a different set of features "
                    "compared with the input schema it was configured with. "
                    "Please check that the inputs passed to the model are only  "
                    "those required by the model. If you're using a Merlin Dataset, "
                    "the `schema` property can be changed to control the features being returned. "
                    f"\nModel expected features:\n\t{model_expected_features}"
                    f"\nCall input features:\n\t{call_input_features}"
                    f"\nFeatures expected by model input schema only:"
                    f"\n\t{model_expected_features.difference(call_input_features)}"
                    f"\nFeatures provided in inputs only:"
                    f"\n\t{call_input_features.difference(model_expected_features)}"
                )

    def _maybe_build(self, inputs):
        if isinstance(inputs, dict):
            self._check_schema_and_inputs_matching(inputs)
            _ragged_inputs = inputs
            if self.prep_features:
                _ragged_inputs = self._prepare_features(inputs)
            feature_shapes = {k: v.shape for k, v in _ragged_inputs.items()}
            feature_dtypes = {k: v.dtype for k, v in _ragged_inputs.items()}

            for block in self.blocks:
                block._feature_shapes = feature_shapes
                block._feature_dtypes = feature_dtypes
                for child in block.submodules:
                    child._feature_shapes = feature_shapes
                    child._feature_dtypes = feature_dtypes
        super()._maybe_build(inputs)

    def build(self, input_shape=None):
        """Builds the model

        Parameters
        ----------
        input_shape : tf.TensorShape, optional
            The input shape, by default None
        """
        last_layer = None

        if self.prep_features:
            self._prepare_features.build(input_shape)
            input_shape = self._prepare_features.compute_output_shape(input_shape)

        if self.pre is not None:
            self.pre.build(input_shape)
            input_shape = self.pre.compute_output_shape(input_shape)

        for layer in self.blocks:
            try:
                layer.build(input_shape)
            except TypeError:
                t, v, tb = sys.exc_info()
                if isinstance(input_shape, dict) and isinstance(last_layer, TabularBlock):
                    v = TypeError(
                        f"Couldn't build {layer}, "
                        f"did you forget to add aggregation to {last_layer}?"
                    )
                six.reraise(t, v, tb)
            input_shape = layer.compute_output_shape(input_shape)
            last_layer = layer

        if self.post is not None:
            self.post.build(input_shape)

        self.built = True

    def call(self, inputs, targets=None, training=False, testing=False, output_context=False):
        outputs = inputs
        features = self._prepare_features(inputs, targets=targets)
        if isinstance(features, tuple):
            features, targets = features
        if self.prep_features:
            outputs = features
        context = self._create_context(
            features,
            targets=targets,
            training=training,
            testing=testing,
        )
        if self.pre:
            outputs, context = self._call_child(self.pre, outputs, context)

        for block in self.blocks:
            outputs, context = self._call_child(block, outputs, context)

        if self.post:
            outputs, context = self._call_child(self.post, outputs, context)

        if output_context:
            return outputs, context

        return outputs

    def _create_context(
        self, inputs, targets=None, training=False, testing=False
    ) -> PredictionContext:
        context = PredictionContext(inputs, targets=targets, training=training, testing=testing)

        return context

    def _call_child(
        self,
        child: tf.keras.layers.Layer,
        inputs,
        context: PredictionContext,
    ):
        call_kwargs = context.to_call_dict()

        # Prevent features to be part of signature of model-blocks
        if any(isinstance(sub, ModelBlock) for sub in child.submodules):
            del call_kwargs["features"]

        outputs = call_layer(child, inputs, **call_kwargs)
        if isinstance(outputs, Prediction):
            targets = outputs.targets if outputs.targets is not None else context.targets
            features = outputs.features if outputs.features is not None else context.features
            if isinstance(child, ModelOutput):
                if not (context.training or context.testing):
                    outputs = outputs[0]
            else:
                outputs = outputs[0]
            context = context.with_updates(targets=targets, features=features)

        return outputs, context

    @property
    def first(self):
        return self.blocks[0]

    @property
    def last(self):
        return self.blocks[-1]

    @classmethod
    def from_block(
        cls,
        block: Block,
        schema: Schema,
        input_block: Optional[Block] = None,
        prediction_tasks: Optional[
            Union[
                "PredictionTask",
                List["PredictionTask"],
                "ParallelPredictionBlock",
                "ModelOutputType",
            ]
        ] = None,
        aggregation="concat",
        **kwargs,
    ) -> "Model":
        """Create a model from a `block`

        Parameters
        ----------
        block: Block
            The block to wrap in-between an InputBlock and prediction task(s)
        schema: Schema
            Schema to use for the model.
        input_block: Optional[Block]
            Block to use as input.
        prediction_tasks: Optional[Union[PredictionTask,List[PredictionTask],
                                ParallelPredictionBlock,ModelOutputType]
        The prediction tasks to be used, by default this will be inferred from the Schema.
        For custom prediction tasks we recommending using OutputBlock and blocks based
        on ModelOutput than the ones based in PredictionTask (that will be deprecated).
        """
        if isinstance(block, SequentialBlock) and is_input_block(block.first):
            if input_block is not None:
                raise ValueError("The block already includes an InputBlock")
            input_block = block.first

        _input_block: Block = input_block or InputBlock(schema, aggregation=aggregation, **kwargs)

        prediction_tasks = parse_prediction_blocks(schema, prediction_tasks)

        return cls(_input_block, block, prediction_tasks)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pre = config.pop("pre", None)
        post = config.pop("post", None)
        schema = config.pop("schema", None)
        batch_size = config.pop("batch_size", None)

        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        if pre is not None:
            pre = tf.keras.layers.deserialize(pre, custom_objects=custom_objects)

        if post is not None:
            post = tf.keras.layers.deserialize(post, custom_objects=custom_objects)

        if schema is not None:
            schema = schema_utils.tensorflow_metadata_json_to_schema(schema)

        model = cls(*layers, pre=pre, post=post, schema=schema)

        # For TF/Keras 2.11 calling the model with sample inputs to trigger build
        # so that variable restore works correctly.
        # TODO: review if this needs changing for 2.12
        if (
            saving_lib
            and hasattr(saving_lib, "_SAVING_V3_ENABLED")
            and saving_lib._SAVING_V3_ENABLED.value
        ):
            inputs = model.get_sample_inputs(batch_size=batch_size)
            if inputs:
                model(inputs)

        return model

    def get_sample_inputs(self, batch_size=None):
        batch_size = batch_size or 2
        if self.input_schema is not None:
            inputs = {}
            for column in self.input_schema:
                shape = [batch_size]
                try:
                    dtype = column.dtype.to("tensorflow")
                except ValueError:
                    dtype = tf.float32

                if column.int_domain:
                    maxval = column.int_domain.max
                elif column.float_domain:
                    maxval = column.float_domain.max
                else:
                    maxval = 1

                if column.is_list and column.is_ragged:
                    row_length = (
                        int(column.value_count.max)
                        if column.value_count and column.value_count.max
                        else 3
                    )
                    values = tf.random.uniform(
                        [batch_size * row_length],
                        dtype=dtype,
                        maxval=maxval,
                    )
                    offsets = tf.cumsum([0] + [row_length] * batch_size)
                    inputs[f"{column.name}__values"] = values
                    inputs[f"{column.name}__offsets"] = offsets
                elif column.is_list:
                    row_length = (
                        int(column.value_count.max)
                        if column.value_count and column.value_count.max
                        else 3
                    )
                    inputs[column.name] = tf.random.uniform(
                        shape + [row_length], dtype=dtype, maxval=maxval
                    )
                else:
                    inputs[column.name] = tf.random.uniform(shape, dtype=dtype, maxval=maxval)
            return inputs

    def get_config(self):
        config = maybe_serialize_keras_objects(self, {}, ["pre", "post"])
        config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        for i, layer in enumerate(self.blocks):
            config[i] = tf.keras.utils.serialize_keras_object(layer)
        config["batch_size"] = self._batch_size

        return config

    def _set_save_spec(self, inputs, args=None, kwargs=None):
        # We need to overwrite this in order to fix a Keras-bug in TF<2.9
        super()._set_save_spec(inputs, args, kwargs)

        if version.parse(tf.__version__) < version.parse("2.9.0"):
            # Keras will interpret kwargs like `features` & `targets` as
            # required args, which is wrong. This is a workaround.
            _arg_spec = self._saved_model_arg_spec
            self._saved_model_arg_spec = ([_arg_spec[0][0]], _arg_spec[1])

    @property
    def frozen_blocks(self):
        """
        Get frozen blocks of model, only on which you called freeze_blocks before, the result dose
        not include those blocks frozen in other methods, for example, if you create the embedding
        and set the `trainable` as False, it would not be tracked by this property, but you can also
        call unfreeze_blocks on those blocks.

        Please note that sub-block of self._frozen_blocks is also frozen, but not recorded by this
        variable, because if you want to unfreeze the whole model, you only need to unfreeze blocks
        you froze before (called freeze_blocks before), this function would unfreeze all sub-blocks
        recursively and automatically.

        If you want to get all frozen blocks and sub-blocks of the model:
            `get_sub_blocks(model.frozen_blocks)`
        """
        return list(self._frozen_blocks)

    def freeze_blocks(
        self,
        blocks: Union[Sequence[Block], Sequence[str]],
    ):
        """Freeze all sub-blocks of given blocks recursively. Please make sure to compile the model
        after freezing.

        Important note about layer-freezing: Calling `compile()` on a model is meant to "save" the
        behavior of that model, which means that whether the layer is frozen or not would be
        preserved for the model, so if you want to freeze any layer of the model, please make sure
        to compile it again.

        TODO: Make it work for graph mode. Now if model compile and fit for multiple times with
        graph mode (run_eagerly=True) could raise TensorFlow error. Please find example in
        test_freeze_parallel_block.

        Parameters
        ----------
        blocks : Union[Sequence[Block], Sequence[str]]
            Blocks or names of blocks to be frozen

        Example :
            ```python
            input_block = ml.InputBlockV2(schema)
            layer_1 = ml.MLPBlock([64], name="layer_1")
            layer_2 = ml.MLPBlock([1], no_activation_last_layer=True, name="layer_2")
            two_layer = ml.SequentialBlock([layer_1, layer_2], name="two_layers")
            body = input_block.connect(two_layer)
            model = ml.Model(body, ml.BinaryClassificationTask("click"))

            # Compile(Make sure set run_eagerly mode) and fit -> model.freeze_blocks -> compile and
            # fit Set run_eagerly=True in order to avoid error: "Called a function referencing
            # variables which have been deleted". Model needs to be built by fit or build.

            model.compile(run_eagerly=True, optimizer=tf.keras.optimizers.SGD(lr=0.1))
            model.fit(ecommerce_data, batch_size=128, epochs=1)

            model.freeze_blocks(["user_categories", "layer_2"])
            # From the result of model.summary(), you can find which block is frozen (trainable: N)
            print(model.summary(expand_nested=True, show_trainable=True, line_length=80))

            model.compile(run_eagerly=False, optimizer="adam")
            model.fit(ecommerce_data, batch_size=128, epochs=10)
            ```

        """
        if not isinstance(blocks, (list, tuple)):
            blocks = [blocks]
        if isinstance(blocks[0], str):
            blocks_to_freeze = self.get_blocks_by_name(blocks)
        elif isinstance(blocks[0], Block):
            blocks_to_freeze = blocks
        for b in blocks_to_freeze:
            b.trainable = False
        self._frozen_blocks.update(blocks_to_freeze)

    def unfreeze_blocks(
        self,
        blocks: Union[Sequence[Block], Sequence[str]],
    ):
        """
        Unfreeze all sub-blocks of given blocks recursively

        Important note about layer-freezing: Calling `compile()` on a model is meant to "save" the
        behavior of that model, which means that whether the layer is frozen or not would be
        preserved for the model, so if you want to freeze any layer of the model, please make sure
        to compile it again.
        """
        if not isinstance(blocks, (list, tuple)):
            blocks = [blocks]
        if isinstance(blocks[0], Block):
            blocks_to_unfreeze = set(get_sub_blocks(blocks))
        elif isinstance(blocks[0], str):
            blocks_to_unfreeze = self.get_blocks_by_name(blocks)
        for b in blocks_to_unfreeze:
            if b not in self._frozen_blocks:
                warnings.warn(
                    f"Block or sub-block {b} was not frozen when calling unfreeze_block("
                    f"{blocks})."
                )
            else:
                self._frozen_blocks.remove(b)
            b.trainable = True

    def unfreeze_all_frozen_blocks(self):
        """
        Unfreeze all blocks (including blocks and sub-blocks) of this model recursively

        Important note about layer-freezing: Calling `compile()` on a model is meant to "save" the
        behavior of that model, which means that whether the layer is frozen or not would be
        preserved for the model, so if you want to freeze any layer of the model, please make sure
        to compile it again.
        """
        for b in self._frozen_blocks:
            b.trainable = True
        self._frozen_blocks = set()

    def get_blocks_by_name(self, block_names: Sequence[str]) -> List[Block]:
        """Get blocks by given block_names, return a list of blocks
        Traverse(Iterate) the model to check each block (sub_block) by BFS"""
        result_blocks = set()
        if not isinstance(block_names, (list, tuple)):
            block_names = [block_names]

        for block in self.blocks:
            # Traversse all submodule (BFS) except ModelContext
            deque = collections.deque()
            if not isinstance(block, ModelContext):
                deque.append(block)
            while deque:
                current_module = deque.popleft()
                # Already found all blocks
                if len(block_names) == 0:
                    break
                # Found a block
                if current_module.name in block_names:
                    result_blocks.add(current_module)
                    block_names.remove(current_module.name)
                for sub_module in current_module._flatten_modules(
                    include_self=False, recursive=False
                ):
                    # Filter out modelcontext
                    if type(sub_module) != ModelContext:
                        deque.append(sub_module)
            if len(block_names) > 0:
                raise ValueError(f"Cannot find block with the name of {block_names}")
        return list(result_blocks)


@runtime_checkable
class RetrievalBlock(Protocol):
    def query_block(self) -> Block:
        ...

    def item_block(self) -> Block:
        ...


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RetrievalModel(Model):
    """Embedding-based retrieval model."""

    def __init__(self, *args, **kwargs):
        kwargs["prep_features"] = False
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        x=None,
        y=None,
        item_corpus: Optional[Union[merlin.io.Dataset, TopKIndexBlock]] = None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        **kwargs,
    ):
        if item_corpus:
            if getattr(self, "has_item_corpus", None) is False:
                raise Exception(
                    "The model.evaluate() was called before without `item_corpus` argument, "
                    "(which is done internally by model.fit() with `validation_data` set) "
                    "and you cannot use model.evaluate() after with `item_corpus` set "
                    "due to a limitation in graph mode. "
                    "Classes based on RetrievalModel (MatrixFactorizationModel,TwoTowerModel) "
                    "are deprecated and we advice using MatrixFactorizationModelV2 and "
                    "TwoTowerModelV2, where this issue does not happen because the evaluation "
                    "over the item catalog is done separately by using "
                    "`model.to_top_k_encoder().evaluate()."
                )

            from merlin.models.tf.core.index import TopKIndexBlock

            self.has_item_corpus = True

            if isinstance(item_corpus, TopKIndexBlock):
                self.loss_block.pre_eval_topk = item_corpus  # type: ignore
            elif isinstance(item_corpus, merlin.io.Dataset):
                item_corpus = unique_rows_by_features(item_corpus, Tags.ITEM, Tags.ITEM_ID)
                item_block = self.retrieval_block.item_block()

                if not getattr(self, "pre_eval_topk", None):
                    topk_metrics = filter_topk_metrics(self.metrics)
                    if len(topk_metrics) == 0:
                        # TODO: Decouple the evaluation of RetrievalModel from the need of using
                        # at least one TopkMetric (how to infer the k for TopKIndexBlock?)
                        raise ValueError(
                            "RetrievalModel evaluation requires at least "
                            "one TopkMetric (e.g., RecallAt(5), NDCGAt(10))."
                        )
                    self.pre_eval_topk = TopKIndexBlock.from_block(
                        item_block,
                        data=item_corpus,
                        k=tf.reduce_max([metric.k for metric in topk_metrics]),
                        context=self.context,
                        **kwargs,
                    )
                else:
                    self.pre_eval_topk.update_from_block(item_block, item_corpus)
            else:
                raise ValueError(
                    "`item_corpus` must be either a `TopKIndexBlock` or a `Dataset`. ",
                    f"Got {type(item_corpus)}",
                )

            # set cache_query to True in the ItemRetrievalScorer
            from merlin.models.tf import ItemRetrievalTask

            if isinstance(self.prediction_tasks[0], ItemRetrievalTask):
                self.prediction_tasks[0].set_retrieval_cache_query(True)  # type: ignore
        else:
            self.has_item_corpus = False

        return super().evaluate(
            x,
            y,
            batch_size,
            verbose,
            sample_weight,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
            return_dict,
            **kwargs,
        )

    @property
    def retrieval_block(self) -> RetrievalBlock:
        return next(b for b in self.blocks if isinstance(b, RetrievalBlock))

    def query_embeddings(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        query_tag: Union[str, Tags] = Tags.USER,
        query_id_tag: Union[str, Tags] = Tags.USER_ID,
    ) -> merlin.io.Dataset:
        """Export query embeddings from the model.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            Dataset to export embeddings from.
        batch_size : int
            Batch size to use for embedding extraction.
        query_tag: Union[str, Tags], optional
            Tag to use for the query.
        query_id_tag: Union[str, Tags], optional
            Tag to use for the query id.

        Returns
        -------
        merlin.io.Dataset
            Dataset with the user/query features and the embeddings
            (one dim per column in the data frame)
        """
        from merlin.models.tf.utils.batch_utils import QueryEmbeddings

        get_user_emb = QueryEmbeddings(self, batch_size=batch_size)

        dataset = unique_rows_by_features(dataset, query_tag, query_id_tag).to_ddf()
        embeddings = dataset.map_partitions(get_user_emb)

        return merlin.io.Dataset(embeddings)

    def item_embeddings(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        item_tag: Union[str, Tags] = Tags.ITEM,
        item_id_tag: Union[str, Tags] = Tags.ITEM_ID,
    ) -> merlin.io.Dataset:
        """Export item embeddings from the model.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            Dataset to export embeddings from.
        batch_size : int
            Batch size to use for embedding extraction.
        item_tag : Union[str, Tags], optional
            Tag to use for the item.
        item_id_tag : Union[str, Tags], optional
            Tag to use for the item id, by default Tags.ITEM_ID

        Returns
        -------
        merlin.io.Dataset
            Dataset with the item features and the embeddings
            (one dim per column in the data frame)
        """
        from merlin.models.tf.utils.batch_utils import ItemEmbeddings

        get_item_emb = ItemEmbeddings(self, batch_size=batch_size)

        dataset = unique_rows_by_features(dataset, item_tag, item_id_tag).to_ddf()
        embeddings = dataset.map_partitions(get_item_emb)

        return merlin.io.Dataset(embeddings)

    def check_for_retrieval_task(self):
        if not (
            getattr(self, "loss_block", None)
            and getattr(self.loss_block, "set_retrieval_cache_query", None)
        ):
            raise ValueError(
                "Your retrieval model should contain an ItemRetrievalTask "
                "in the end (loss_block)."
            )

    def to_top_k_recommender(
        self,
        item_corpus: Union[merlin.io.Dataset, TopKIndexBlock],
        k: Optional[int] = None,
        **kwargs,
    ) -> ModelBlock:
        """Convert the model to a Top-k Recommender.
        Parameters
        ----------
        item_corpus: Union[merlin.io.Dataset, TopKIndexBlock]
            Dataset to convert to a Top-k Recommender.
        k: int
            Number of recommendations to make.
        Returns
        -------
        SequentialBlock
        """
        import merlin.models.tf as ml

        if isinstance(item_corpus, merlin.io.Dataset):
            if not k:
                topk_metrics = filter_topk_metrics(self.metrics)
                if topk_metrics:
                    k = tf.reduce_max([metric.k for metric in topk_metrics])
                else:
                    raise ValueError("You must specify a k for the Top-k Recommender.")

            data = unique_rows_by_features(item_corpus, Tags.ITEM, Tags.ITEM_ID)
            topk_index = ml.TopKIndexBlock.from_block(
                self.retrieval_block.item_block(), data=data, k=k, **kwargs
            )
        else:
            topk_index = item_corpus
        # Set the blocks for recommenders with built=True to keep pre-trained embeddings
        recommender_block = self.retrieval_block.query_block().connect(topk_index)
        recommender_block.built = True
        recommender = ModelBlock(recommender_block)
        recommender.built = True
        return recommender


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RetrievalModelV2(Model):
    def __init__(
        self,
        *,
        query: Union[Encoder, tf.keras.layers.Layer],
        output: Union[ModelOutput, tf.keras.layers.Layer],
        candidate: Optional[Union[Encoder, tf.keras.layers.Layer]] = None,
        query_name="query",
        candidate_name="candidate",
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        if isinstance(output, ContrastiveOutput):
            query_name = output.query_name
            candidate_name = output.candidate_name

        if query and candidate:
            encoder = ParallelBlock({query_name: query, candidate_name: candidate})
        else:
            encoder = query

        super().__init__(encoder, output, pre=pre, post=post, prep_features=False, **kwargs)

        self._query_name = query_name
        self._candidate_name = candidate_name
        self._encoder = encoder
        self._output = output

    def query_embeddings(
        self,
        dataset: Optional[merlin.io.Dataset] = None,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        query = self.query_encoder if self.has_candidate_encoder else self.encoder

        if dataset is not None and hasattr(query, "encode"):
            return query.encode(dataset, index=index, **kwargs)

        if hasattr(query, "to_dataset"):
            return query.to_dataset(**kwargs)

        return query.encode(dataset, index=index, **kwargs)

    def candidate_embeddings(
        self,
        dataset: Optional[merlin.io.Dataset] = None,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        **kwargs,
    ) -> merlin.io.Dataset:
        if self.has_candidate_encoder:
            candidate = self.candidate_encoder

            if dataset is not None and hasattr(candidate, "encode"):
                return candidate.encode(dataset, index=index, **kwargs)

            if hasattr(candidate, "to_dataset"):
                return candidate.to_dataset(**kwargs)

            return candidate.encode(dataset, index=index, **kwargs)

        if isinstance(self.last, ContrastiveOutput):
            return self.last.to_dataset()

        raise Exception(...)

    @property
    def encoder(self):
        return self._encoder

    @property
    def has_candidate_encoder(self):
        return (
            isinstance(self.encoder, ParallelBlock)
            and self._candidate_name in self.encoder.parallel_dict
        )

    @property
    def query_encoder(self) -> Encoder:
        if self.has_candidate_encoder:
            output = self.encoder[self._query_name]
        else:
            output = self.encoder

        output = self._check_encoder(output)

        return output

    @property
    def candidate_encoder(self) -> Encoder:
        output = None
        if self.has_candidate_encoder:
            output = self.encoder[self._candidate_name]

        if output:
            return self._check_encoder(output)

        raise ValueError("No candidate encoder found.")

    def _check_encoder(self, maybe_encoder):
        output = maybe_encoder

        from merlin.models.tf.core.encoder import Encoder

        if isinstance(output, SequentialBlock):
            output = Encoder(*maybe_encoder.layers)

        if not isinstance(output, Encoder):
            raise ValueError(f"Query encoder should be an Encoder, got {type(output)}")

        return output

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pre = config.pop("pre", None)
        if pre is not None:
            pre = tf.keras.layers.deserialize(pre, custom_objects=custom_objects)

        post = config.pop("post", None)
        if post is not None:
            post = tf.keras.layers.deserialize(post, custom_objects=custom_objects)

        encoder = config.pop("_encoder", None)
        if encoder is not None:
            encoder = tf.keras.layers.deserialize(encoder, custom_objects=custom_objects)

        output = config.pop("_output", None)
        if output is not None:
            output = tf.keras.layers.deserialize(output, custom_objects=custom_objects)

        output = RetrievalModelV2(query=encoder, output=output, pre=pre, post=post)
        output.__class__ = cls

        return output

    def get_config(self):
        config = maybe_serialize_keras_objects(self, {}, ["pre", "post", "_encoder", "_output"])

        return config

    def to_top_k_encoder(
        self,
        candidates: merlin.io.Dataset = None,
        candidate_id=Tags.ITEM_ID,
        strategy: Union[str, tf.keras.layers.Layer] = "brute-force-topk",
        k: int = 10,
        **kwargs,
    ):
        from merlin.models.tf.core.encoder import TopKEncoder

        """Method to get a top-k encoder

        Parameters
        ----------
        candidate : merlin.io.Dataset, optional
            Dataset of unique candidates, by default None
        candidate_id:
            Column to use as the candidates index,
            by default Tags.ITEM_ID
        strategy: str
            Strategy to use for retrieving the top-k candidates of
            a given query, by default brute-force-topk
        """
        candidates_embeddings = self.candidate_embeddings(candidates, index=candidate_id, **kwargs)
        topk_model = TopKEncoder(
            self.query_encoder,
            topk_layer=strategy,
            k=k,
            candidates=candidates_embeddings,
            target=self.encoder._schema.select_by_tag(candidate_id).first.name,
        )
        return topk_model


def _maybe_convert_merlin_dataset(data, batch_size, shuffle=True, **kwargs):
    # Check if merlin-dataset is passed
    if hasattr(data, "to_ddf"):
        if not batch_size:
            raise ValueError("batch_size must be specified when using merlin-dataset.")

        data = Loader(data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        if not shuffle:
            kwargs.pop("shuffle", None)

    return data


def get_task_names_from_outputs(
    outputs: Union[List[str], List[PredictionTask], ParallelPredictionBlock, List[ParallelBlock]]
):
    "Extracts tasks names from outputs"
    if isinstance(outputs, ParallelPredictionBlock):
        output_names = outputs.task_names
    elif isinstance(outputs, ParallelBlock):
        if all(isinstance(x, ModelOutput) for x in outputs.parallel_values):
            output_names = [o.task_name for o in outputs.parallel_values]
        else:
            raise ValueError("The blocks within ParallelBlock must be ModelOutput.")
    elif isinstance(outputs, (list, tuple)):
        if all(isinstance(x, PredictionTask) for x in outputs):
            output_names = [o.task_name for o in outputs]  # type: ignore
        elif all(isinstance(x, ModelOutput) for x in outputs):
            output_names = [o.task_name for o in outputs]  # type: ignore
        else:
            raise ValueError(
                "The blocks within the list/tuple must be ModelOutput or PredictionTask."
            )
    elif isinstance(outputs, PredictionTask):
        output_names = [outputs.task_name]
    elif isinstance(outputs, ModelOutput):
        output_names = [outputs.task_name]
    else:
        raise ValueError("Invalid model outputs")
    return output_names
