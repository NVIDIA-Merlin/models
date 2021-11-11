import logging
from typing import List, Optional, Sequence, Text, Type, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from tensorflow.keras.layers import Layer

from ..core import Block, PredictionTask
from ..features.embedding import EmbeddingFeatures


def name_fn(name, inp):
    return "/".join([name, inp]) if name else None


MetricOrMetricClass = Union[tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric]]

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy()
    DEFAULT_METRICS = (
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.AUC,
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )
        self.pre = tf.keras.layers.Dense(1, activation="sigmoid", name=self.child_name("logit"))


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RegressionTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.MeanSquaredError()
    DEFAULT_METRICS = (tf.keras.metrics.RootMeanSquaredError,)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )
        self.pre = tf.keras.layers.Dense(1, name=self.child_name("logit"))


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SampledItemPredictionTask(PredictionTask):
    DEFAULT_METRICS = tuple()

    def __init__(
        self,
        schema: Schema,
        dim: int,
        num_sampled: int,
        item_id: Optional[str] = None,
        target_name: Optional[str] = str(Tag.ITEM_ID),
        task_name: str = "item-prediction",
        metrics: Optional[List[MetricOrMetricClass]] = DEFAULT_METRICS,
        pre: Optional[Layer] = None,
        task_block: Optional[Layer] = None,
        prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            None,
            task_name=task_name,
            metrics=metrics,
            pre=pre,
            task_block=task_block,
            prediction_metrics=prediction_metrics,
            label_metrics=label_metrics,
            loss_metrics=loss_metrics,
            target_name=target_name,
            name=name,
            **kwargs,
        )
        _schema = schema.select_by_name(item_id) if item_id else schema.select_by_tag(Tag.ITEM_ID)
        self.item_embedding = EmbeddingFeatures.from_schema(
            _schema, embedding_dim_default=dim, item_id=item_id
        )
        self.num_sampled = num_sampled

    def build_task(self, input_shape, schema: Schema, body: Block, **kwargs):
        return super().build(input_shape)

    def build(self, input_shape):
        self.item_embedding.build(input_shape)
        self.num_classes = self.item_embedding.item_embedding_table.shape[0]
        self.zero_bias = self.add_weight(
            shape=(self.num_classes,),
            initializer=tf.keras.initializers.Zeros,
            dtype=tf.float32,
            trainable=False,
            name="bias",
        )

        return super().build(input_shape)

    def compute_loss(
        self,
        predictions,
        targets,
        training: bool = False,
        compute_metrics=True,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]
        if isinstance(predictions, dict) and self.target_name:
            predictions = predictions[self.task_name]

        targets = tf.one_hot(targets, self.num_classes)

        loss = tf.expand_dims(
            tf.nn.sampled_softmax_loss(
                weights=self.item_embedding.item_embedding_table,
                biases=self.zero_bias,
                labels=targets,
                inputs=predictions,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                num_true=self.num_classes,
            ),
            axis=1,
        )

        if compute_metrics:
            update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

            update_ops = [x for x in update_ops if x is not None]

            with tf.control_dependencies(update_ops):
                return tf.identity(loss)

        return loss
