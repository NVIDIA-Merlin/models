from typing import Tuple, Optional, List, Text, Dict

import tensorflow as tf
from merlin_standard_lib import Tag, Schema
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import array_ops

from merlin_models.tf.core import (
    PredictionBlock,
    PredictionTask,
    Sampler,
    MetricOrMetricClass,
    Block,
    prediction_transforms_registry
)


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class ItemPredictionTask(PredictionTask):
    DEFAULT_LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    DEFAULT_METRICS = ()

    # TODO: Move these metrics from T4Rec
    # DEFAULT_METRICS = (
    #     # default metrics suppose labels are int encoded
    #     NDCGAt(top_ks=[10, 20], labels_onehot=True),
    #     AvgPrecisionAt(top_ks=[10, 20], labels_onehot=True),
    #     RecallAt(top_ks=[10, 20], labels_onehot=True),
    # )

    def __init__(
            self,
            schema: Schema,
            loss=DEFAULT_LOSS,
            metrics=DEFAULT_METRICS,
            target_name: Optional[str] = None,
            task_name: Optional[str] = None,
            task_block: Optional[Layer] = None,
            weight_tying: bool = False,
            softmax_temperature: float = 1,
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
        self.weight_tying = weight_tying
        self.num_classes = schema.categorical_cardinalities()[str(Tag.ITEM_ID)]
        self.softmax_temperature = softmax_temperature
        self.loss = loss

    def build(self, input_shape):
        if self.weight_tying:
            self.item_embedding_table = self.context.get_embedding(Tag.ITEM_ID)
            self.bias = self.add_weight(
                name="output_layer_bias",
                shape=(self.num_classes,),
                initializer=tf.keras.initializers.Zeros(),
            )
        else:
            self.output_layer = Dense(
                units=self.num_classes,
                kernel_initializer="random_normal",
                bias_initializer="zeros",
                name="logits",
            )
            self.item_embedding_table = self.output_layer.kernel
            self.bias = self.output_layer.bias
        return super().build(input_shape)

    def _compute_loss(self, predictions, targets, sample_weight=None, training: bool = False,
                      **kwargs) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        if self.weight_tying:
            logits = tf.matmul(inputs, tf.transpose(self.item_embedding_table))
            logits = tf.nn.bias_add(logits, self.bias)
        else:
            logits = self.output_layer(inputs)

        if self.softmax_temperature:
            # Softmax temperature to reduce model overconfidence
            # and better calibrate probs and accuracy
            logits = logits / self.softmax_temperature

        predictions = tf.nn.log_softmax(logits, axis=-1)

        return predictions

    def metric_results(self, mode: str = None) -> Dict[str, tf.Tensor]:
        metrics = {metric.name: metric.result() for metric in self.eval_metrics}
        topks = {metric.name: metric.top_ks for metric in self.eval_metrics}
        # explode metrics for each cut-off in top_ks
        results = {}
        for name, metric in metrics.items():
            for measure, k in zip(metric, topks[name]):
                results[f"{name}_{k}"] = measure

        return results


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class SampledItemPredictionTask(ItemPredictionTask):
    def __init__(self,
                 schema: Schema,
                 num_sampled: int,
                 loss=ItemPredictionTask.DEFAULT_LOSS,
                 metrics=ItemPredictionTask.DEFAULT_METRICS,
                 target_name: Optional[str] = None, task_name: Optional[str] = None,
                 task_block: Optional[Layer] = None, weight_tying: bool = False,
                 softmax_temperature: float = 1, **kwargs):
        super().__init__(schema, loss, metrics, target_name, task_name, task_block, weight_tying,
                         softmax_temperature, **kwargs)
        self.num_sampled = num_sampled

    def _compute_loss(self, predictions, targets, sample_weight=None, training: bool = False,
                      **kwargs) -> tf.Tensor:
        if training:
            loss = tf.expand_dims(
                tf.nn.sampled_softmax_loss(
                    weights=self.item_embedding_table,
                    biases=self.bias,
                    labels=targets,
                    inputs=predictions,
                    num_sampled=self.num_sampled,
                    num_classes=self.num_classes,
                    num_true=self.num_classes,
                ),
                axis=1,
            )
        else:
            labels = tf.one_hot(targets, self.n_classes)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions)

        return loss

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            return inputs

        logits = tf.matmul(inputs, tf.transpose(self.item_embedding_table))
        logits = tf.nn.bias_add(logits, self.bias)

        return logits


@prediction_transforms_registry.register_with_multiple_names("sampling-bias-correction")
class SamplingBiasCorrection(PredictionBlock):
    def __init__(self, bias_feature_name: str = "popularity", **kwargs):
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.bias_feature_name = bias_feature_name

    def transform(
            self,
            predictions,
            targets,
            training=True,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        sampling_bias = self.context.tensors.get(self.bias_feature_name)
        if sampling_bias is not None:
            predictions -= tf.math.log(sampling_bias)
        else:
            # TODO : add warning
            pass

        return predictions, targets


class InBatchNegativeSampling(PredictionBlock):
    def transform(
            self,
            predictions,
            targets,
            training=True,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        scores = tf.linalg.matmul(*list(predictions.values()), transpose_b=True)

        if targets is not None:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            targets = tf.eye(*scores.shape)

        return scores, targets


@prediction_transforms_registry.register_with_multiple_names("negative-sampling")
class NegativeSampling(PredictionBlock):
    def __init__(self, *sampler: Sampler, in_batch=True, **kwargs):
        self.sampler = sampler
        self.in_batch_sampler = InBatchNegativeSampling() if in_batch else None

        if not in_batch and not sampler:
            raise ValueError("Either in_batch or sampler must be set")

        super(NegativeSampling, self).__init__(**kwargs)

    def sample(self) -> tf.Tensor:
        if len(self.sampler) > 1:
            return tf.concat([sampler.sample() for sampler in self.sampler], axis=0)

        return self.sampler[0].sample()

    def transform(
            self,
            predictions,
            targets,
            training=True,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.in_batch_sampler:
            predictions, targets = self.in_batch_sampler(predictions, targets)

        if self.sampler:
            extra_negatives: tf.Tensor = self.sample()
            predictions = tf.concat([predictions, extra_negatives], axis=0)
            targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        return predictions, targets


# TODO: Implement this for the MIND model: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(PredictionBlock):
    def transform(
            self,
            predictions,
            targets,
            training=True,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")

#
# class RetrievalPredictionTask(PredictionTask):
#     def __init__(self,
#                  loss: Optional[tf.keras.losses.Loss] = None,
#                  in_batch_negatives: bool = True,
#                  extra_negatives: Optional[Sampler] = None,
#                  target_name: Optional[str] = None,
#                  task_name: Optional[str] = None,
#                  metrics: Optional[List[MetricOrMetricClass]] = None, pre: Optional[Layer] = None,
#                  prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
#                  label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
#                  loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
#                  name: Optional[Text] = None, **kwargs) -> None:
#         loss = loss if loss is not None else CategoricalCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
#         super().__init__(loss, target_name, task_name, metrics, pre, None, prediction_metrics,
#                          label_metrics, loss_metrics, name, **kwargs)
#         self.in_batch_negatives = in_batch_negatives
#         self.extra_negatives = extra_negatives
#
#     def compute_loss(
#             self,
#             predictions,
#             targets,
#             training: bool = False,
#             compute_metrics=True,
#             sample_weight: Optional[tf.Tensor] = None,
#             **kwargs
#     ) -> tf.Tensor:
#         if isinstance(targets, dict) and self.target_name:
#             targets = targets[self.target_name]
#         if isinstance(predictions, dict) and self.target_name:
#             predictions = predictions[self.task_name]
#
#         if self.in_batch_negatives:
#             norm_vecs = [tf.linalg.l2_normalize(inp, axis=1) for inp in list(predictions.values())]
#             scores = tf.linalg.matmul(*norm_vecs, transpose_b=True)
#
#             if targets is not None:
#                 if len(targets.shape) == 2:
#                     targets = tf.squeeze(targets)
#                 targets = tf.linalg.diag(targets)
#             else:
#                 targets = tf.eye(tf.shape(scores)[0], tf.shape(scores)[1])
#         else:
#             if targets is None:
#                 raise ValueError("Targets are required when in-batch negative sampling is disabled")
#             scores = tf.keras.layers.Dot(axes=1, normalize=True)(list(predictions.values()))
#
#         if self.extra_negatives:
#             extra_negatives: tf.Tensor = self.extra_negatives.sample()
#             extra_negatives = array_ops.stop_gradient(extra_negatives,
#                                                       name="extra_negatives_stop_gradient")
#             scores = tf.concat([scores, extra_negatives], axis=0)
#             targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)
#
#         # Sampling bias correction
#         # TODO: add popularity to standard tags
#         popularity = self.get_from_context("popularity")
#         if popularity is not None:
#             scores -= tf.math.log(popularity)
#
#         loss = self.loss(y_true=targets, y_pred=scores, sample_weight=sample_weight)
#
#         if compute_metrics:
#             update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)
#
#             update_ops = [x for x in update_ops if x is not None]
#
#             with tf.control_dependencies(update_ops):
#                 return tf.identity(loss)
#
#         return loss
