from __future__ import annotations

import collections
import inspect
import sys
import warnings
from collections.abc import Sequence as SequenceCollection
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

import six
import tensorflow as tf
from keras.utils.losses_utils import cast_losses_to_common_dtype
from packaging import version
from tensorflow.keras.utils import unpack_x_y_sample_weight

import merlin.io
from merlin.models.tf.core.base import Block, ModelContext, PredictionOutput, is_input_block
from merlin.models.tf.core.combinators import SequentialBlock
from merlin.models.tf.core.prediction import Prediction, PredictionContext
from merlin.models.tf.core.tabular import TabularBlock
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.loader import Loader
from merlin.models.tf.losses.base import loss_registry
from merlin.models.tf.metrics.topk import TopKMetricsAggregator, filter_topk_metrics, split_metrics
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.outputs.base import ModelOutput
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.transforms.tensor import ListToRagged
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.search_utils import find_all_instances_in_layers
from merlin.models.tf.utils.tf_utils import (
    call_layer,
    get_sub_blocks,
    maybe_serialize_keras_objects,
)
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Schema, Tags

if TYPE_CHECKING:
    from merlin.models.tf.core.index import TopKIndexBlock


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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ModelBlock(Block, tf.keras.Model):
    """Block that extends `tf.keras.Model` to make it saveable."""

    def __init__(self, block: Block, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        if hasattr(self, "set_schema"):
            block_schema = getattr(block, "schema", None)
            self.set_schema(block_schema)

    def call(self, inputs, **kwargs):
        if "features" not in kwargs:
            kwargs["features"] = inputs
        outputs = call_layer(self.block, inputs, **kwargs)
        return outputs

    def build(self, input_shapes):
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
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
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
            loss: Loss function. Maybe be a string (name of loss function), or
              a `tf.keras.losses.Loss` instance. See `tf.keras.losses`. A loss
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
            metrics: List of metrics to be evaluated by the model during training
              and testing. Each of this can be a string (name of a built-in
              function), function or a `tf.keras.metrics.Metric` instance. See
              `tf.keras.metrics`. Typically you will use `metrics=['accuracy']`. A
              function is any callable with the signature `result = fn(y_true,
              y_pred)`. To specify different metrics for different outputs of a
              multi-output model, you could also pass a dictionary, such as
              `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
              You can also pass a list to specify a metric or a list of metrics
              for each output, such as `metrics=[['accuracy'], ['accuracy', 'mse']]`
              or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
              strings 'accuracy' or 'acc', we convert this to one of
              `tf.keras.metrics.BinaryAccuracy`,
              `tf.keras.metrics.CategoricalAccuracy`,
              `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss
              function used and the model output shape. We do a similar
              conversion for the strings 'crossentropy' and 'ce' as well.
            loss_weights: Optional list or dictionary specifying scalar coefficients
              (Python floats) to weight the loss contributions of different model
              outputs. The loss value that will be minimized by the model will then
              be the *weighted sum* of all individual losses, weighted by the
              `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping to the model's
                  outputs. If a dict, it is expected to map output names (strings)
                  to scalar coefficients.
            weighted_metrics: List of metrics to be evaluated and weighted by
              `sample_weight` or `class_weight` during training and testing.
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

        # Initializing model control flags controlled by MetricsComputeCallback()
        self._should_compute_train_metrics_for_batch = tf.Variable(
            dtype=tf.bool,
            name="should_compute_train_metrics_for_batch",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            initial_value=lambda: False,
        )

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

        super(BaseModel, self).compile(
            optimizer=optimizer,
            loss=self._create_loss(loss),
            metrics=self._create_metrics(metrics),
            weighted_metrics=self._create_weighted_metrics(weighted_metrics),
            run_eagerly=run_eagerly,
            loss_weights=loss_weights,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            from_serialized=from_serialized,
            **kwargs,
        )

    def _create_metrics(self, metrics=None):
        out = {}

        num_v1_blocks = len(self.prediction_tasks)
        if isinstance(metrics, dict):
            out = metrics

        elif isinstance(metrics, (list, tuple)):
            # Retrieve top-k metrics & wrap them in TopKMetricsAggregator
            topk_metrics, topk_aggregators, other_metrics = split_metrics(metrics)
            if len(topk_metrics) > 0:
                topk_aggregators.append(TopKMetricsAggregator(*topk_metrics))
            metrics = other_metrics + topk_aggregators

            if num_v1_blocks > 0:
                if num_v1_blocks == 1:
                    out[self.prediction_tasks[0].task_name] = metrics
                else:
                    for i, task in enumerate(self.prediction_tasks):
                        out[task.task_name] = metrics[i]
            else:
                if len(self.model_outputs) == 1:
                    out[self.model_outputs[0].full_name] = metrics
                else:
                    for i, block in enumerate(self.model_outputs):
                        out[block.full_name] = metrics[i]

        elif metrics is None:
            for task_name, task in self.prediction_tasks_by_name().items():
                out[task_name] = [m() if inspect.isclass(m) else m for m in task.DEFAULT_METRICS]

            for prediction_name, prediction_block in self.outputs_by_name().items():
                out[prediction_name] = prediction_block.default_metrics_fn()
                if len(self.model_outputs) > 1:
                    for metric in out[prediction_name]:
                        metric._name = "/".join([prediction_block.full_name, metric.name])
        else:
            out = metrics

        return out

    def _create_weighted_metrics(self, weighted_metrics=None):
        out = {}

        num_v1_blocks = len(self.prediction_tasks)

        if isinstance(weighted_metrics, dict):
            out = weighted_metrics

        elif isinstance(weighted_metrics, (list, tuple)):
            if num_v1_blocks > 0:
                if num_v1_blocks == 1:
                    out[self.prediction_tasks[0].task_name] = weighted_metrics
                else:
                    for i, task in enumerate(self.prediction_tasks):
                        out[task.task_name] = weighted_metrics[i]
            else:
                if len(self.model_outputs) == 1:
                    out[self.model_outputs[0].full_name] = weighted_metrics
                else:
                    for i, block in enumerate(self.model_outputs):
                        out[block.full_name] = weighted_metrics[i]

        return out

    def _create_loss(self, loss=None):
        out = {}

        if isinstance(loss, (tf.keras.losses.Loss, str)):
            if len(self.prediction_tasks) == 1:
                out = {task.task_name: loss for task in self.prediction_tasks}
            elif len(self.model_outputs) == 1:
                out = {task.name: loss for task in self.model_outputs}

        # If loss is not provided, use the defaults from the prediction-tasks.
        if not loss:
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
        Prediction (v2) or PredictionOutput (v1 - depreciated) objects

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

        return Prediction(predictions, targets, sample_weights)

    def train_step(self, data):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            x, y, sample_weight = unpack_x_y_sample_weight(data)
            outputs = self.call_train_test(x, y, sample_weight=sample_weight, training=True)
            loss = self.compute_loss(x, outputs.targets, outputs.predictions, outputs.sample_weight)

        self._validate_target_and_loss(outputs.targets, loss)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        metrics = self.compute_metrics(outputs, training=True)

        # Adding regularization loss to metrics
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))

        return metrics

    def test_step(self, data):
        """Custom test step using the `compute_loss` method."""

        x, y, sample_weight = unpack_x_y_sample_weight(data)
        outputs = self.call_train_test(x, y, sample_weight=sample_weight, testing=True)

        if getattr(self, "pre_eval_topk", None) is not None:
            # During eval, the retrieval-task only returns positive scores
            # so we need to retrieve top-k negative scores to compute the loss
            outputs = self.pre_eval_topk.call_outputs(outputs)

        self.compute_loss(x, outputs.targets, outputs.predictions, outputs.sample_weight)

        metrics = self.compute_metrics(outputs, training=False)

        # Adding regularization loss to metrics
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))

        return metrics

    @tf.function
    def compute_metrics(
        self,
        prediction_outputs: PredictionOutput,
        training: bool,
    ) -> Dict[str, tf.Tensor]:
        """Overrides Model.compute_metrics() for some custom behaviour
           like compute metrics each N steps during training
           and allowing to feed additional information required by specific metrics

        Parameters
        ----------
        prediction_outputs : PredictionOutput
            Contains properties with targets and predictions
        training : bool
            Flag that indicates if metrics are being computed during
            training or evaluation

        Returns
        -------
        Dict[str, tf.Tensor]
            Dict with the metrics values
        """

        should_compute_metrics = self._should_compute_train_metrics_for_batch or not training
        if should_compute_metrics:
            # This ensures that compiled metrics are built
            # to make self.compiled_metrics.metrics available
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(
                    prediction_outputs.predictions, prediction_outputs.targets
                )

            # Providing label_relevant_counts for TopkMetrics, as metric.update_state()
            # should have standard signature for better compatibility with Keras methods
            # like self.compiled_metrics.update_state()
            if hasattr(prediction_outputs, "label_relevant_counts"):
                for topk_metric in filter_topk_metrics(self.compiled_metrics.metrics):
                    topk_metric.label_relevant_counts = prediction_outputs.label_relevant_counts

            self.compiled_metrics.update_state(
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

        # Bind schema from dataset to model in case we can't infer it from the inputs
        if isinstance(x, Loader):
            self.schema = x.schema

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

    def _add_metrics_callback(self, callbacks, train_metrics_steps):
        if callbacks is None:
            callbacks = []

        if isinstance(callbacks, SequenceCollection):
            callbacks = list(callbacks)
        else:
            callbacks = [callbacks]

        callback_types = [type(callback) for callback in callbacks]
        if MetricsComputeCallback not in callback_types:
            # Adding a callback to control metrics computation
            callbacks.append(MetricsComputeCallback(train_metrics_steps))

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
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, shuffle=False, **kwargs)

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
        **kwargs,
    ):
        x = _maybe_convert_merlin_dataset(x, batch_size, shuffle=False, **kwargs)

        return super(BaseModel, self).predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Model(BaseModel):
    def __init__(
        self,
        *blocks: Block,
        context: Optional[ModelContext] = None,
        pre: Optional[tf.keras.layers.Layer] = None,
        post: Optional[tf.keras.layers.Layer] = None,
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

        input_block_schemas = [
            block.schema for block in self.submodules if getattr(block, "is_input", False)
        ]
        self.schema = sum(input_block_schemas, Schema())
        self._frozen_blocks = set()

    def _maybe_build(self, inputs):
        if isinstance(inputs, dict):
            _ragged_inputs = ListToRagged()(inputs)
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
        context = self._create_context(
            ListToRagged()(inputs),
            targets=targets,
            training=training,
            testing=testing,
        )

        outputs = inputs
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
            Union["PredictionTask", List["PredictionTask"], "ParallelPredictionBlock"]
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
        prediction_tasks: Optional[
            Union[PredictionTask, List[PredictionTask], ParallelPredictionBlock]
        ]
            Prediction tasks to use.
        """
        if isinstance(block, SequentialBlock) and is_input_block(block.first):
            if input_block is not None:
                raise ValueError("The block already includes an InputBlock")
            input_block = block.first

        _input_block: Block = input_block or InputBlock(schema, aggregation=aggregation, **kwargs)

        prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)

        return cls(_input_block, block, prediction_tasks)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pre = config.pop("pre", None)
        post = config.pop("post", None)
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        if pre is not None:
            pre = tf.keras.layers.deserialize(pre, custom_objects=custom_objects)

        if post is not None:
            post = tf.keras.layers.deserialize(post, custom_objects=custom_objects)

        return cls(*layers, pre=pre, post=post)

    def get_config(self):
        config = maybe_serialize_keras_objects(self, {}, ["pre", "post"])
        for i, layer in enumerate(self.blocks):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

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
        self.has_item_corpus = False

        if item_corpus:
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


def _maybe_convert_merlin_dataset(data, batch_size, shuffle=True, **kwargs):
    # Check if merlin-dataset is passed
    if hasattr(data, "to_ddf"):
        if not batch_size:
            raise ValueError("batch_size must be specified when using merlin-dataset.")

        data = Loader(data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        if not shuffle:
            kwargs.pop("shuffle", None)

    return data
