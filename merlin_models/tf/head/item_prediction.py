from typing import Dict, Optional, Tuple

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from tensorflow.python.keras.layers import Dense
from tensorflow.python.layers.base import Layer

from merlin_models.tf.core import (
    PredictionBlock,
    PredictionTask,
    Sampler,
    prediction_block_registry,
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
        pre_call: Optional[PredictionBlock] = None,
        pre_loss: Optional[PredictionBlock] = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre_call=pre_call,
            pre_loss=pre_loss,
            **kwargs,
        )
        self.weight_tying = weight_tying
        self.num_classes = schema.categorical_cardinalities()[str(Tag.ITEM_ID)]
        self.softmax_temperature = softmax_temperature
        self.loss = loss

    def build(self, input_shape):
        if self.weight_tying:
            self.output_layer_kernel = self.context.get_embedding(Tag.ITEM_ID)
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
            self.output_layer_kernel = self.output_layer.kernel
            self.bias = self.output_layer.bias
        return super().build(input_shape)

    def _compute_loss(
        self, predictions, targets, sample_weight=None, training: bool = False, **kwargs
    ) -> tf.Tensor:
        return self.loss(targets, predictions, sample_weight=sample_weight)

    def call(self, inputs, training=False, **kwargs):
        if self.weight_tying:
            logits = tf.matmul(inputs, tf.transpose(self.output_layer_kernel))
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
    def __init__(
        self,
        schema: Schema,
        num_sampled: int,
        loss=ItemPredictionTask.DEFAULT_LOSS,
        metrics=ItemPredictionTask.DEFAULT_METRICS,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        pre_call: Optional[PredictionBlock] = None,
        pre_loss: Optional[PredictionBlock] = None,
        **kwargs,
    ):
        super().__init__(
            schema,
            loss,
            metrics,
            target_name,
            task_name,
            task_block,
            weight_tying,
            softmax_temperature,
            pre_call=pre_call,
            pre_loss=pre_loss,
            **kwargs,
        )
        self.num_sampled = num_sampled

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            return inputs

        logits = tf.matmul(inputs, tf.transpose(self.item_embedding_table))
        logits = tf.nn.bias_add(logits, self.bias)

        return logits


@tf.keras.utils.register_keras_serializable(package="transformers4rec")
class ItemRetrievalTask(ItemPredictionTask):
    def __init__(
        self,
        schema: Schema,
        loss=ItemPredictionTask.DEFAULT_LOSS,
        metrics=ItemPredictionTask.DEFAULT_METRICS,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        pre_call: Optional[PredictionBlock] = "negative-sampling",
        pre_loss: Optional[PredictionBlock] = None,
        normalize=True,
        **kwargs,
    ):
        super().__init__(
            schema,
            loss,
            metrics,
            target_name,
            task_name,
            task_block,
            weight_tying,
            softmax_temperature,
            pre_call=pre_call,
            pre_loss=pre_loss,
            **kwargs,
        )
        self.normalize = normalize

    def build(self, input_shape):
        return Layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        if self.normalize:
            inputs = [tf.linalg.l2_normalize(inp, axis=1) for inp in list(inputs.values())]
        predictions = tf.linalg.matmul(*inputs, transpose_b=True)

        return predictions


@prediction_block_registry.register_with_multiple_names("sampling-bias-correction")
class SamplingBiasCorrection(PredictionBlock):
    def __init__(self, bias_feature_name: str = "popularity", **kwargs):
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.bias_feature_name = bias_feature_name

    def predict(self, predictions, targets, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        sampling_bias = self.context.tensors.get(self.bias_feature_name)
        if sampling_bias is not None:
            predictions -= tf.math.log(sampling_bias)
        else:
            # TODO : add warning
            pass

        return predictions, targets


class InBatchNegativeSampling(PredictionBlock):
    def predict(self, predictions, targets, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        predictions = tf.linalg.matmul(*list(predictions.values()), transpose_b=True)

        if targets is not None:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            targets = tf.eye(*predictions.shape)

        return predictions, targets


@prediction_block_registry.register_with_multiple_names("negative-sampling")
class NegativeSampling(PredictionBlock):
    def __init__(self, *sampler: Sampler, in_batch=True, **kwargs):
        self.sampler = sampler
        self.in_batch_neg_sampler = InBatchNegativeSampling() if in_batch else None

        if not in_batch and not sampler:
            raise ValueError("Either in_batch or sampler must be set")

        super(NegativeSampling, self).__init__(**kwargs)

    def sample(self) -> tf.Tensor:
        if len(self.sampler) > 1:
            return tf.concat([sampler.sample() for sampler in self.sampler], axis=0)

        return self.sampler[0].sample()

    def predict(self, predictions, targets, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.in_batch_neg_sampler:
            predictions, targets = self.in_batch_neg_sampler(predictions, targets)

        if self.sampler:
            extra_negatives: tf.Tensor = self.sample()
            predictions = tf.concat([predictions, extra_negatives], axis=0)
            targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        return predictions, targets


# TODO: Implement this for the MIND model: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(PredictionBlock):
    def predict(self, predictions, targets, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")
