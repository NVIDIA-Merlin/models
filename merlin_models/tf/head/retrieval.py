from typing import Tuple, Optional, List, Text

import tensorflow as tf
from merlin_standard_lib import Tag, Schema
from tensorflow.python.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import array_ops

from merlin_models.tf.features.embedding import EmbeddingFeatures
from merlin_models.tf.core import (
    PredictionTransformation,
    PredictionTask,
    Sampler,
    MetricOrMetricClass,
    Block, prediction_transforms_registry
)


@prediction_transforms_registry.register_with_multiple_names("sampling-bias-correction")
class SamplingBiasCorrection(PredictionTransformation):
    def __init__(self, bias_feature_name: str = "popularity", **kwargs):
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.bias_feature_name = bias_feature_name

    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        sampling_bias = self.get_from_context(self.bias_feature_name)
        if sampling_bias is not None:
            predictions -= tf.math.log(sampling_bias)

        return predictions, targets


class _InBatchNegativeSampling(PredictionTransformation):
    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        scores = tf.linalg.matmul(*list(predictions.values()), transpose_b=True)

        if targets is not None:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            targets = tf.eye(*scores.shape)

        return scores, targets


@prediction_transforms_registry.register_with_multiple_names("negative-sampling")
class NegativeSampling(PredictionTransformation):
    def __init__(self, *sampler: Sampler, in_batch=True, **kwargs):
        self.sampler = sampler
        self.in_batch_sampler = _InBatchNegativeSampling() if in_batch else None

        if not in_batch and not sampler:
            raise ValueError("Either in_batch or sampler must be set")

        super(NegativeSampling, self).__init__(**kwargs)

    def sample(self) -> tf.Tensor:
        if len(self.sampler) > 1:
            return tf.concat([sampler.sample() for sampler in self.sampler], axis=0)

        return self.sampler[0].sample()

    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.in_batch_sampler:
            predictions, targets = self.in_batch_sampler(predictions, targets)

        if self.sampler:
            extra_negatives: tf.Tensor = self.sample()
            # extra_negatives = array_ops.stop_gradient(extra_negatives,
            #                                           name="extra_negatives_stop_gradient")
            predictions = tf.concat([predictions, extra_negatives], axis=0)
            targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        return predictions, targets


# TODO: Implement this for the MIND model: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(PredictionTransformation):
    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")


class RetrievalPredictionTask(PredictionTask):
    def __init__(self,
                 loss: Optional[tf.keras.losses.Loss] = None,
                 in_batch_negatives: bool = True,
                 extra_negatives: Optional[Sampler] = None,
                 target_name: Optional[str] = None,
                 task_name: Optional[str] = None,
                 metrics: Optional[List[MetricOrMetricClass]] = None, pre: Optional[Layer] = None,
                 prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 name: Optional[Text] = None, **kwargs) -> None:
        loss = loss if loss is not None else CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        super().__init__(loss, target_name, task_name, metrics, pre, None, prediction_metrics,
                         label_metrics, loss_metrics, name, **kwargs)
        self.in_batch_negatives = in_batch_negatives
        self.extra_negatives = extra_negatives

    def compute_loss(
            self,
            predictions,
            targets,
            training: bool = False,
            compute_metrics=True,
            sample_weight: Optional[tf.Tensor] = None,
            **kwargs
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]
        if isinstance(predictions, dict) and self.target_name:
            predictions = predictions[self.task_name]

        if self.in_batch_negatives:
            norm_vecs = [tf.linalg.l2_normalize(inp, axis=1) for inp in list(predictions.values())]
            scores = tf.linalg.matmul(*norm_vecs, transpose_b=True)

            if targets is not None:
                if len(targets.shape) == 2:
                    targets = tf.squeeze(targets)
                targets = tf.linalg.diag(targets)
            else:
                targets = tf.eye(tf.shape(scores)[0], tf.shape(scores)[1])
        else:
            if targets is None:
                raise ValueError("Targets are required when in-batch negative sampling is disabled")
            scores = tf.keras.layers.Dot(axes=1, normalize=True)(list(predictions.values()))

        if self.extra_negatives:
            extra_negatives: tf.Tensor = self.extra_negatives.sample()
            extra_negatives = array_ops.stop_gradient(extra_negatives,
                                                      name="extra_negatives_stop_gradient")
            scores = tf.concat([scores, extra_negatives], axis=0)
            targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        # Sampling bias correction
        # TODO: add popularity to standard tags
        popularity = self.get_from_context("popularity")
        if popularity is not None:
            scores -= tf.math.log(popularity)

        loss = self.loss(y_true=targets, y_pred=scores, sample_weight=sample_weight)

        if compute_metrics:
            update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

            update_ops = [x for x in update_ops if x is not None]

            with tf.control_dependencies(update_ops):
                return tf.identity(loss)

        return loss


class SampledSoftmax(Layer):
    def __init__(self, num_classes: int, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.num_classes = num_classes

    def call(self, y_true, query_embeddings, item_embeddings, context):
        return tf.expand_dims(
            tf.nn.sampled_softmax_loss(
                weights=query_embeddings,
                biases=self.zero_bias,
                labels=y_true,
                inputs=item_embeddings,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                num_true=self.num_classes,
            ),
            axis=1,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SampledItemPredictionTask(PredictionTask):
    DEFAULT_METRICS = tuple()

    def __init__(
        self,
        schema: Schema,
        num_sampled: int,
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
        self.num_classes = schema.categorical_cardinalities()[target_name]
        self.num_sampled = num_sampled

    def build_task(self, input_shape, schema: Schema, body: Block, **kwargs):
        return super().build(input_shape)

    def build(self, input_shape):
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
