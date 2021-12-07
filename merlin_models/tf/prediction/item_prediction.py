#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.layers.base import Layer

from merlin_models.tf.core import Block, PredictionTask, Sampler
from merlin_standard_lib import Schema, Tag

from .classification import MultiClassClassificationTask
from .ranking_metric import ranking_metrics


@Block.registry.register_with_multiple_names("sampling-bias-correction")
class SamplingBiasCorrection(Block):
    def __init__(self, bias_feature_name: str = "popularity", **kwargs):
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.bias_feature_name = bias_feature_name

    def call_features(self, features, **kwargs):
        self.bias = features[self.bias_feature_name]

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        inputs -= tf.math.log(self.bias)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class SoftmaxTemperature(Block):
    def __init__(self, temperature: float, **kwargs):
        super(SoftmaxTemperature, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        return inputs / self.temperature

    def compute_output_shape(self, input_shape):
        return input_shape


class ItemSoftmaxWeightTying(Block):
    def __init__(self, schema: Schema, bias_initializer="zeros", **kwargs):
        super(ItemSoftmaxWeightTying, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        self.num_classes = schema.categorical_cardinalities()[str(Tag.ITEM_ID)]

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )
        return super().build(input_shape)

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        embedding_table = self.context.get_embedding(Tag.ITEM_ID)
        logits = tf.matmul(inputs, embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        predictions = tf.nn.log_softmax(logits, axis=-1)

        return predictions


@Block.registry.register_with_multiple_names("in-batch-negative-sampling")
class InBatchNegativeSampling(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        assert len(inputs) == 2
        if training:
            return tf.linalg.matmul(*list(inputs.values()), transpose_b=True)

        return self.dot(list(inputs.values()))

    def call_targets(self, predictions, targets, **kwargs) -> tf.Tensor:
        if targets is not None:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            targets = tf.eye(*predictions.shape)

        return targets

    def compute_output_shape(self, input_shape):
        return input_shape


class ExtraNegativeSampling(Block):
    def __init__(self, *sampler: Sampler, **kwargs):
        self.sampler = sampler
        super(ExtraNegativeSampling, self).__init__(**kwargs)

    def sample(self) -> tf.Tensor:
        if len(self.sampler) > 1:
            return tf.concat([sampler.sample() for sampler in self.sampler], axis=0)

        return self.sampler[0].sample()

    def call(self, inputs, training=True, **kwargs):
        if training:
            extra_negatives: tf.Tensor = self.sample()
            self.extra_negatives_shape = extra_negatives.shape
            inputs = tf.concat([inputs, extra_negatives], axis=0)

        return inputs

    def call_targets(self, inputs, targets, training=True, **kwargs) -> tf.Tensor:
        if training:
            targets = tf.concat([targets, tf.zeros(self.extra_negatives_shape)], axis=0)

        return targets


class L2Norm(Block):
    # TODO: Make this a TabularTransformation
    #  & allow for standard blocks to be used in the prediction-path.
    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)

    def predict(self, inputs, targets=None, training=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, dict):
            inputs = {key: tf.linalg.l2_normalize(inp, axis=1) for key, inp in inputs.items()}
        else:
            inputs = tf.linalg.l2_normalize(inputs, axis=1)

        return inputs, targets

    def compute_output_shape(self, input_shape):
        return input_shape


# TODO: Implement this for the MIND prediction: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(Block):
    def predict(
        self, predictions, targets=None, training=True, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")


def ItemRetrievalTask(
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    ),
    metrics=ranking_metrics(top_ks=[10, 20]),
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    softmax_temperature: float = 1,
    normalize: bool = True,
) -> MultiClassClassificationTask:
    """
    Function to create the ItemRetrieval task with the right parameters.

    Parameters
    ----------
        loss: tf.keras.losses.Loss
            Loss function.
            Defaults to `tf.keras.losses.CategoricalCrossentropy()`.
        metrics: Sequence[MetricOrMetricClass]
            List of top-k ranking metrics.
            Defaults to MultiClassClassificationTask.DEFAULT_METRICS["ranking"].
        extra_pre_call: Optional[PredictionBlock]
            Optional extra pre-call block. Defaults to None.
        target_name: Optional[str]
            If specified, name of the target tensor to retrieve from dataloader.
            Defaults to None.
        task_name: Optional[str]
            name of the task.
            Defaults to None.
        task_block: Block
            The `Block` that applies additional layers op to inputs.
            Defaults to None.
        softmax_temperature: float
            Parameter used to reduce model overconfidence, so that softmax(logits / T).
            Defaults to 1.
        normalize: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to True.

    Returns
    -------
        PredictionTask
            The item retrieval prediction task
    """
    pre_call = InBatchNegativeSampling()

    if normalize:
        pre_call = L2Norm().connect(pre_call)

    if softmax_temperature != 1:
        pre_call = pre_call.connect(SoftmaxTemperature(softmax_temperature))

    if extra_pre_call is not None:
        pre_call = pre_call.connect(extra_pre_call)

    return MultiClassClassificationTask(
        target_name,
        task_name,
        task_block,
        loss=loss,
        metrics=metrics,
        pre_call=pre_call,
    )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemPredictionTask(PredictionTask):
    DEFAULT_LOSS = SparseCategoricalCrossentropy(from_logits=True)
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
        pre: Optional[Block] = None,
        **kwargs,
    ):
        super().__init__(
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
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
            logits = tf.matmul(inputs, self.output_layer_kernel, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.bias)
        else:
            logits = self.output_layer(inputs)

        if self.softmax_temperature:
            # Softmax temperature to reduce prediction overconfidence
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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
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
        pre: Optional[Block] = None,
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
            pre=pre,
            **kwargs,
        )
        self.num_sampled = num_sampled

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            return inputs

        logits = tf.matmul(inputs, tf.transpose(self.item_embedding_table))
        logits = tf.nn.bias_add(logits, self.bias)

        return logits
