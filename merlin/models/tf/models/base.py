from __future__ import annotations

import inspect
from collections.abc import Sequence as SequenceCollection
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Union, runtime_checkable

import tensorflow as tf
from keras.utils.losses_utils import cast_losses_to_common_dtype
from tensorflow.python.keras.engine import data_adapter

import merlin.io
from merlin.models.config.schema import FeatureCollection
from merlin.models.tf.blocks.core.base import Block, ModelContext, PredictionOutput, is_input_block
from merlin.models.tf.blocks.core.combinators import SequentialBlock
from merlin.models.tf.blocks.core.context import FeatureContext
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.inputs.base import InputBlock
from merlin.models.tf.losses.base import loss_registry
from merlin.models.tf.metrics.ranking import RankingMetric
from merlin.models.tf.models.utils import parse_prediction_tasks
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.utils.search_utils import find_all_instances_in_layers
from merlin.models.tf.utils.tf_utils import call_layer
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Schema, Tags

if TYPE_CHECKING:
    from merlin.models.tf.blocks.core.index import TopKIndexBlock


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

    def call(self, inputs, **kwargs):
        outputs = self.block(inputs, **kwargs)
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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Model(tf.keras.Model):
    def __init__(
        self,
        *blocks: Block,
        context: Optional[ModelContext] = None,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        context = context or ModelContext()
        if len(blocks) == 1 and isinstance(blocks[0], SequentialBlock):
            self.block = blocks[0]
        else:
            self.block = SequentialBlock(blocks, context=context, block_name="blocks")
        if not getattr(self.block, "_context", None):
            self.block._set_context(context)
        self.context = context
        self._is_fitting = False

        input_block_schemas = [
            block.schema for block in self.block.submodules if getattr(block, "is_input", False)
        ]
        self.schema = sum(input_block_schemas, Schema())

        self.as_dense = AsDenseFeatures()

        # Initializing model control flags controlled by MetricsComputeCallback()
        self._should_compute_train_metrics_for_batch = tf.Variable(
            dtype=tf.bool,
            name="should_compute_train_metrics_for_batch",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            initial_value=lambda: False,
        )

    def call(self, inputs, **kwargs):
        if not kwargs.get("feature_context", None):
            features = FeatureCollection(self.schema, self.as_dense(inputs))
            kwargs["feature_context"] = FeatureContext(features)

        outputs = call_layer(self.block, inputs, **kwargs)
        return outputs

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

        self.output_names = [task.task_name for task in self.prediction_tasks]

        _metrics = {}
        if isinstance(metrics, (list, tuple)) and len(self.prediction_tasks) == 1:
            _metrics = {task.task_name: metrics for task in self.prediction_tasks}

        # If metrics are not provided, use the defaults from the prediction-tasks.
        # TODO: Do the same for weight_metrics.
        if not metrics:
            for task_name, task in self.prediction_tasks_by_name().items():
                _metrics[task_name] = [
                    m() if inspect.isclass(m) else m for m in task.DEFAULT_METRICS
                ]

        _loss = {}
        if isinstance(loss, (tf.keras.losses.Loss, str)) and len(self.prediction_tasks) == 1:
            _loss = {task.task_name: loss for task in self.prediction_tasks}

        # If loss is not provided, use the defaults from the prediction-tasks.
        if not loss:
            for task_name, task in self.prediction_tasks_by_name().items():
                _loss[task_name] = task.DEFAULT_LOSS

        for key in _loss:
            if isinstance(_loss[key], str) and _loss[key] in loss_registry:
                _loss[key] = loss_registry.parse(_loss[key])

        super(Model, self).compile(
            optimizer=optimizer,
            loss=_loss,
            metrics=_metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            loss_weights=loss_weights,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )

    @property
    def first(self):
        return self.block.layers[0]

    @property
    def last(self):
        return self.block.layers[-1]

    @property
    def prediction_tasks(self) -> List[PredictionTask]:
        from merlin.models.tf.prediction_tasks.base import PredictionTask

        results = find_all_instances_in_layers(self.block, PredictionTask)

        return results

    def prediction_tasks_by_name(self) -> Dict[str, PredictionTask]:
        return {task.task_name: task for task in self.prediction_tasks}

    def prediction_tasks_by_target(self) -> Dict[str, List[PredictionTask]]:
        outputs: Dict[str, Union[PredictionTask, List[PredictionTask]]] = {}
        for task in self.prediction_tasks:
            if task.target in outputs:
                if isinstance(outputs[task.target], list):
                    outputs[task.target].append(task)
                else:
                    outputs[task.target] = [outputs[task.target], task]
            outputs[task.target] = task

        return outputs

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
        if is_input_block(block.first):
            if input_block is not None:
                raise ValueError("The block already includes an InputBlock")
            input_block = block.first

        _input_block: Block = input_block or InputBlock(schema, aggregation=aggregation, **kwargs)

        prediction_tasks = parse_prediction_tasks(schema, prediction_tasks)

        return cls(_input_block, block, prediction_tasks)

    def prediction_output(
        self, x, y=None, training=False, testing=False, **kwargs
    ) -> PredictionOutput:
        features = FeatureCollection(self.schema, self.as_dense(x))
        feature_context = FeatureContext(features)

        forward = self(
            x,
            targets=y,
            training=training,
            testing=testing,
            feature_context=feature_context,
            **kwargs,
        )
        predictions, targets, output = {}, {}, None
        for task in self.prediction_tasks:
            task_x = forward
            if isinstance(forward, dict) and task.task_name in forward:
                task_x = forward[task.task_name]
            if isinstance(task_x, PredictionOutput):
                output = task_x
                task_y = task_x.targets
                task_x = task_x.predictions
            else:
                task_y = y[task.target_name] if isinstance(y, dict) and y else y

            targets[task.task_name] = task_y
            predictions[task.task_name] = task_x

        if len(predictions) == 1 and len(targets) == 1:
            predictions = predictions[list(predictions.keys())[0]]
            targets = targets[list(targets.keys())[0]]

        if output:
            return output.copy_with_updates(predictions, targets)

        return PredictionOutput(predictions, targets)

    def train_step(self, data):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
            outputs = self.prediction_output(x, y, training=True)
            loss = self.compute_loss(x, outputs.targets, outputs.predictions, sample_weight)

        self._validate_target_and_loss(outputs.targets, loss)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        metrics = self.compute_metrics(
            x, outputs.targets, outputs.predictions, sample_weight, training=True
        )
        # Adding regularization loss to metrics
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))

        return metrics

    def test_step(self, data):
        """Custom test step using the `compute_loss` method."""

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        outputs = self.prediction_output(x, y, testing=True)

        if getattr(self, "pre_eval_topk", None) is not None:
            # During eval, the retrieval-task only returns positive scores
            # so we need to retrieve top-k negative scores to compute the loss
            outputs = self.pre_eval_topk.call_outputs(outputs)

        self.compute_loss(x, outputs.targets, outputs.predictions, sample_weight)
        metrics = self.compute_metrics(
            x, outputs.targets, outputs.predictions, sample_weight, training=False
        )
        # Adding regularization loss to metrics
        metrics["regularization_loss"] = tf.reduce_sum(cast_losses_to_common_dtype(self.losses))

        return metrics

    @tf.function
    def compute_metrics(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor,
        training: bool,
    ) -> Dict[str, tf.Tensor]:
        """Overrides default Keras Model.compute_metrics()
           to compute metrics each N steps (and not for all steps)
           during training

        Parameters
        ----------
        x : tf.Tensor
            Input features. Not used, and kept only to align with the signature
            of super Model.compute_metrics()
        y : tf.Tensor
            Target values
        y_pred : tf.Tensor
            Predictions values
        sample_weight : tf.Tensor
            Sample weights for weighted metrics
        training : bool
            Flag that indicates if metrics are being computed during
            training or evaluation

        Returns
        -------
        Dict[str, tf.Tensor]
            Dict with the metrics values
        """
        del x
        # During training updates metric states each N steps
        if self._should_compute_train_metrics_for_batch or not training:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
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

        return super(Model, self).predict(
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

    @classmethod
    def from_config(cls, config, custom_objects=None):
        block = tf.keras.utils.deserialize_keras_object(config.pop("block"))

        return cls(block, **config)

    def get_config(self):
        return {"block": tf.keras.utils.serialize_keras_object(self.block)}


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
        self.has_ranking_metric = any(isinstance(m, RankingMetric) for m in self.metrics)
        self.has_item_corpus = False

        if item_corpus:
            from merlin.models.tf.blocks.core.index import TopKIndexBlock

            self.has_item_corpus = True

            if isinstance(item_corpus, TopKIndexBlock):
                self.loss_block.pre_eval_topk = item_corpus  # type: ignore
            elif isinstance(item_corpus, merlin.io.Dataset):
                item_corpus = unique_rows_by_features(item_corpus, Tags.ITEM, Tags.ITEM_ID)
                item_block = self.retrieval_block.item_block()

                if not getattr(self, "pre_eval_topk", None):
                    ranking_metrics = list(
                        [metric for metric in self.metrics if isinstance(metric, RankingMetric)]
                    )
                    self.pre_eval_topk = TopKIndexBlock.from_block(
                        item_block,
                        data=item_corpus,
                        k=tf.reduce_max([metric.k for metric in ranking_metrics]),
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
        return next(b for b in self.block if isinstance(b, RetrievalBlock))

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
                ranking_metrics = list(
                    [metric for metric in self.metrics if isinstance(metric, RankingMetric)]
                )
                if ranking_metrics:
                    k = tf.reduce_max([metric.k for metric in ranking_metrics])
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

        data = BatchedDataset(data, batch_size=batch_size, shuffle=shuffle, **kwargs)

        if not shuffle:
            kwargs.pop("shuffle", None)

    return data
