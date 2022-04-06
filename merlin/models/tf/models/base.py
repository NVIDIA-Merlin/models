from collections import Sequence as SequenceCollection
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

import tensorflow as tf

import merlin.io
from merlin.models.tf.blocks.core.base import Block, BlockContext
from merlin.models.tf.blocks.core.combinators import SequentialBlock
from merlin.models.tf.metrics.ranking import RankingMetric
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.mixins import LossMixin, MetricsMixin, ModelLikeBlock
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Schema, Tags


class MetricsComputeCallback(tf.keras.callbacks.Callback):
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

    def on_test_begin(self, logs=None):
        self.model._should_compute_eval_metrics_as_in_training.assign(self._is_fitting)

    def on_test_end(self, logs=None):
        self.model._should_compute_eval_metrics_as_in_training.assign(False)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ModelBlock(Block, tf.keras.Model):
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
class Model(tf.keras.Model, LossMixin, MetricsMixin):
    def __init__(
        self,
        *blocks: Union[Block, ModelLikeBlock],
        context: Optional[BlockContext] = None,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        context = context or BlockContext()
        if (
            len(blocks) == 1
            and isinstance(blocks[0], SequentialBlock)
            and isinstance(blocks[0].layers[-1], ModelLikeBlock)
        ):
            self.block = blocks[0]
        else:
            if not isinstance(blocks[-1], ModelLikeBlock):
                raise ValueError("Last block must be able to calculate loss & metrics.")
            self.block = SequentialBlock(blocks, context=context)
        if not getattr(self.block, "_context", None):
            self.block._set_context(context)
        self.context = context
        self._is_fitting = False

        # Initializing model control flags controlled by MetricsComputeCallback()
        self._should_compute_train_metrics_for_batch = tf.Variable(
            dtype=tf.bool,
            name="should_compute_train_metrics_for_batch",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            initial_value=lambda: False,
        )
        self._should_compute_eval_metrics_as_in_training = tf.Variable(
            dtype=tf.bool,
            name="should_compute_eval_metrics_as_in_training",
            trainable=False,
            synchronization=tf.VariableSynchronization.NONE,
            initial_value=lambda: False,
        )

    def call(self, inputs, **kwargs):
        outputs = self.block(inputs, **kwargs)
        return outputs

    # @property
    # def inputs(self):
    #     return self.block.inputs

    @property
    def first(self):
        return self.block.layers[0]

    @property
    def last(self):
        return self.block.layers[-1]

    @property
    def loss_block(self) -> ModelLikeBlock:
        return self.block.last if isinstance(self.block, SequentialBlock) else self.block

    @property
    def schema(self) -> Schema:
        return self.block.schema

    @classmethod
    def from_block(
        cls,
        block: Block,
        schema: Schema,
        input_block: Optional[Block] = None,
        prediction_tasks: Optional[
            Union["PredictionTask", List["PredictionTask"], "ParallelPredictionBlock"]
        ] = None,
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

        return block.to_model(
            schema, input_block=input_block, prediction_tasks=prediction_tasks, **kwargs
        )

    def compute_loss(
        self,
        inputs: Union[tf.Tensor, TabularData],
        targets: Union[tf.Tensor, TabularData],
        compute_metrics=False,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        return self.loss_block.compute_loss(
            inputs, targets, training=training, compute_metrics=compute_metrics, **kwargs
        )

    def calculate_metrics(
        self,
        outputs,
        mode: str = "val",
        forward: bool = True,
        training: bool = False,
        **kwargs,
    ) -> Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]:
        return self.loss_block.calculate_metrics(
            outputs=outputs, mode=mode, forward=forward, training=training, **kwargs
        )

    def metric_results(self, mode=None):
        return self.loss_block.metric_results(mode=mode)

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            if isinstance(inputs, tuple):
                if len(inputs) == 1:
                    inputs = inputs[0]
                    targets = None
                else:
                    inputs, targets = inputs
            else:
                targets = None

            predictions = self(inputs, training=True)
            loss = self.compute_loss(
                predictions,
                targets,
                training=True,
                compute_metrics=self._should_compute_train_metrics_for_batch,
            )
            tf.assert_rank(
                loss,
                0,
                "The loss tensor should have rank 0. "
                "Check if you are using a tf.keras.losses.Loss with 'reduction' "
                "properly set",
            )
            assert loss.dtype == tf.float32, (
                f"The loss dtype should be tf.float32 but is rather {loss.dtype}. "
                "Ensure that your model output has tf.float32 dtype, as "
                "that should be the case when using mixed_float16 policy "
                "to avoid numerical instabilities."
            )

            regularization_loss = tf.reduce_sum(self.losses)

            total_loss = tf.add_n([loss, regularization_loss])

            if getattr(self.optimizer, "get_scaled_loss", False):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # If mixed precision (mixed_float16 policy) is enabled
        # (and the optimizer is automatically wrapped by
        #  tensorflow.keras.mixed_precision.LossScaleOptimizer())
        if getattr(self.optimizer, "get_scaled_loss", False):
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = self.loss_block.metric_result_dict()

        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        if isinstance(inputs, tuple):
            if len(inputs) == 1:
                inputs = inputs[0]
                targets = None
            else:
                inputs, targets = inputs
        else:
            targets = None

        def compute_loss_metrics(training):
            predictions = self(inputs, training=training)
            loss = self.compute_loss(predictions, targets, training=training, compute_metrics=True)
            tf.assert_rank(
                loss,
                0,
                "The loss tensor should have rank 0. "
                "Check if you are using a tf.keras.losses.Loss with 'reduction' "
                "properly set",
            )
            return loss

        loss = tf.cond(
            self._should_compute_eval_metrics_as_in_training,
            lambda: compute_loss_metrics(training=True),
            lambda: compute_loss_metrics(training=False),
        )

        # Casting regularization loss to fp16 if needed to match the main loss
        regularization_loss = tf.cast(tf.reduce_sum(self.losses), loss.dtype)

        total_loss = loss + regularization_loss

        metrics = self.loss_block.metric_result_dict()
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

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
        # Check if merlin-dataset is passed
        if hasattr(x, "to_ddf"):
            if not batch_size:
                raise ValueError("batch_size must be specified when using merlin-dataset.")
            from merlin.models.tf.dataset import BatchedDataset

            x = BatchedDataset(x, batch_size=batch_size, **kwargs)

        if hasattr(validation_data, "to_ddf"):
            if not batch_size:
                raise ValueError("batch_size must be specified when using merlin-dataset.")
            from merlin.models.tf.dataset import BatchedDataset

            kwargs.pop("shuffle", None)
            validation_data = BatchedDataset(
                validation_data, batch_size=batch_size, shuffle=False, **kwargs
            )

        callbacks = self._add_metrics_callback(callbacks, train_metrics_steps)

        return super().fit(
            x,
            y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

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
        # Check if merlin-dataset is passed
        if hasattr(x, "to_ddf"):
            if not batch_size:
                raise ValueError("batch_size must be specified when using merlin-dataset.")
            from merlin.models.tf.dataset import BatchedDataset

            x = BatchedDataset(x, batch_size=batch_size, shuffle=False, **kwargs)

        # Load BruteForce top-k evaluation
        # using evaluation_candidates as indices.
        if isinstance(self, RetrievalModel):
            self.check_for_retrieval_task()
            # We need to load the top-k indices at each call of the evaluate()
            # to ensure most updated pre-computed embeddings are loaded.
            self = self._load_or_update_candidates_index()

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


class RetrievalModel(Model):
    """Embedding-based retrieval model."""

    def __init__(
        self,
        *blocks: Union[Block, ModelLikeBlock],
        context: Optional[BlockContext] = None,
        evaluation_candidates: merlin.io.Dataset = None,
        **kwargs,
    ):
        super().__init__(*blocks, context=context, **kwargs)

        if not any(isinstance(b, RetrievalBlock) for b in self.block):
            raise ValueError("Model must contain a `RetrievalBlock`.")

        self.evaluation_candidates = evaluation_candidates

    @property
    def retrieval_block(self) -> RetrievalBlock:
        return next(b for b in self.block if isinstance(b, RetrievalBlock))

    def query_embeddings(
        self,
        dataset: merlin.io.Dataset,
        query_tag=Tags.USER,
        query_id_tag=Tags.USER_ID,
        batch_size=None,
    ) -> merlin.io.Dataset:
        """Export query embeddings from the model.
        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to export embeddings from.
        query_tag: str
            Tag to use for the query.
        query_id_tag: str
            Tag to use for the query id.
        batch_size: int
            Batch size to use for embedding extraction.
        Returns
        -------
        merlin.io.Dataset
        """
        from merlin.models.tf.utils.batch_utils import QueryEmbeddings

        get_user_emb = QueryEmbeddings(self, batch_size=batch_size)

        dataset = unique_rows_by_features(dataset, query_tag, query_id_tag).to_ddf()
        embeddings = dataset.map_partitions(get_user_emb)

        return merlin.io.Dataset(embeddings)

    def item_embeddings(
        self,
        dataset: merlin.io.Dataset,
        item_tag=Tags.ITEM,
        item_id_tag=Tags.ITEM_ID,
        batch_size=None,
    ) -> merlin.io.Dataset:
        """Export item embeddings from the model.
        Parameters
        ----------
        dataset: merlin.io.Dataset
            Dataset to export embeddings from.
        item_tag: str
            Tag to use for the item.
        item_id_tag: str
            Tag to use for the item id.
        batch_size: int
            Batch size to use for embedding extraction.
        Returns
        -------
        merlin.io.Dataset
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

    def set_retrieval_candidates_for_evaluation(self, candidates: merlin.io.Dataset):
        self.evaluation_candidates = unique_rows_by_features(candidates, Tags.ITEM, Tags.ITEM_ID)

        ranking_metrics = list(
            [metric for metric in self.loss_block.eval_metrics if isinstance(metric, RankingMetric)]
        )
        self._k = tf.reduce_max([metric.k for metric in ranking_metrics])

    def _load_or_update_candidates_index(self, **kwargs):
        """Update the model with a top-k evaluation block."""
        import merlin.models.tf as ml

        self.check_for_retrieval_task()
        if self.evaluation_candidates is None:
            raise ValueError(
                "You need to specify the set of negatives to use for evaluation "
                "via `set_retrieval_candidates_for_evaluation` method"
            )

        if self.loss_block.pre_eval_topk is None:
            topk_index = ml.TopKIndexBlock.from_block(
                self.retrieval_block.item_block(),
                data=self.evaluation_candidates,
                k=self._k,
                context=self.context,
                **kwargs,
            )
            self.loss_block.pre_eval_topk = topk_index  # type: ignore
        else:
            if not self._should_compute_eval_metrics_as_in_training:
                self.loss_block.pre_eval_topk.update_from_block(
                    self.retrieval_block.item_block(), self.evaluation_candidates
                )

        # set cache_query to True in the ItemRetrievalScorer
        self.loss_block.set_retrieval_cache_query(True)  # type: ignore

        return self

    def to_top_k_recommender(self, data: merlin.io.Dataset, k: int, **kwargs) -> ModelBlock:
        """Convert the model to a Top-k Recommender.
        Parameters
        ----------
        data: merlin.io.Dataset
            Dataset to convert to a Top-k Recommender.
        k: int
            Number of recommendations to make.
        Returns
        -------
        SequentialBlock
        """
        import merlin.models.tf as ml

        topk_index = ml.TopKIndexBlock.from_block(
            self.retrieval_block.item_block(), data=data, k=k, **kwargs
        )
        recommender = self.retrieval_block.query_block().connect(topk_index)

        return ModelBlock(recommender)
