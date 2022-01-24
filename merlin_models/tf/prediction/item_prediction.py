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
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops.nn_impl import _compute_sampled_logits

from merlin_models.tf.block.transformations import L2Norm
from merlin_models.tf.core import Block, Sampler
from merlin_standard_lib import Schema, Tag

from ..block.aggregation import SequenceAggregation, SequenceAggregator
from ..block.inputs import InputBlock
from ..block.mlp import MLPBlock
from .classification import MultiClassClassificationTask, Softmax
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


class SampledSoftmax(Block):
    """
        Compute the softmax scores on a subset of sampled candidates to optimize
        training. During inference, the scores are computed over the whole
        catalog of items.

        Reference of the method can be found at [Jean et al., 2014](http://arxiv.org/abs/1412.2007)

        We use the default log-uniform sampler given by tensorflow:
        [log_uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler)

        We note that this default sampler requires that item-ids are encoded based
        on a decreasing order of their count frequency and that the classes' expected counts
        are approximated based on their index order.

    Parameters:
    -----------
        schema: Schema
        num_sampled: int
            The number of candidates to sample during training
        sampler
            The function to sample a subset of classes based on a given distribution,
            it returns a tuple of
            (sampled ids, expected_count of true classes, expected_count of negative ones)
            Defaults to tf.random.log_uniform_candidate_sampler
        num_true: int
            The number of target classes per training example
            Defaults to 1
        remove_accidental_hits: bool
            Ignore sampled items that are equal to the target classes
            Defaults to True
        weight_tying: bool
            The item_id embedding weights are shared with the prediction output layer.
            Defaults to True
        bias_initializer: str
            Initializer for setting the initial random biases
            Defaults to 'zeros'
        kernel_initializer: str
            Initializer for setting the initial random weights of output layer if
            `weight_tying=False`
            Defaults to 'glorot_uniform'
        seed: int
            Fix the random values returned by the sampler to ensure reproducibility
            Defaults to None

    Returns:
    -------
        targets, logits: tf.Tensor, tf.Tensor
            During training, return the concatenated tensor of true class
            and sampled negatives of shape (bs, num_sampled+1), as well as the related logits.
            During evaluation, returns the input tensor of true class, and the related logits.
    """

    def __init__(
        self,
        schema: Schema,
        num_sampled: int,
        sampler=tf.random.log_uniform_candidate_sampler,
        num_true: int = 1,
        remove_accidental_hits: bool = True,
        weight_tying: bool = True,
        bias_initializer: str = "zeros",
        kernel_initializer: str = "glorot_uniform",
        seed: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_sampled = num_sampled
        self.sampler = sampler
        self.num_true = num_true
        self.remove_accidental_hits = remove_accidental_hits
        self.weight_tying = weight_tying
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.num_classes = schema.categorical_cardinalities()[str(Tag.ITEM_ID)]
        self.seed = seed

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )
        if not self.weight_tying:
            self.item_embedding_weights = self.add_weight(
                name="output_layer_weights",
                shape=(self.num_classes, input_shape[-1]),
                initializer=self.kernel_initializer,
            )
        return super().build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            return inputs

        if self.weight_tying:
            weights = self.context.get_embedding(Tag.ITEM_ID)
        else:
            weights = self.item_embedding_weights
        logits = tf.matmul(inputs, tf.transpose(weights))
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

    def call_targets(self, predictions, targets, training=True, **kwargs) -> tf.Tensor:
        if training:
            if self.weight_tying:
                weights = self.context.get_embedding(Tag.ITEM_ID)
            else:
                weights = self.item_embedding_weights

            if targets.dtype != tf.int64:
                targets = tf.cast(targets, tf.int64)

            sampled_values = self.sampler(
                true_classes=tf.reshape(targets, (-1, 1)),
                num_true=self.num_true,
                num_sampled=self.num_sampled,
                unique=True,
                range_max=self.num_classes,
                seed=self.seed,
            )

            logits, targets = _compute_sampled_logits(
                weights=weights,
                biases=self.bias,
                labels=tf.reshape(targets, (-1, 1)),
                inputs=predictions,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                num_true=self.num_true,
                sampled_values=sampled_values,
                subtract_log_q=True,
                remove_accidental_hits=self.remove_accidental_hits,
            )
        return targets, logits


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
        if targets:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            num_rows, num_columns = tf.shape(predictions)[0], tf.shape(predictions)[1]
            targets = tf.eye(num_rows, num_columns)

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

    def call_targets(self, predictions, targets, training=True, **kwargs):
        if training:
            targets = tf.concat([targets, tf.zeros(self.extra_negatives_shape)], axis=0)

        return targets


# TODO: Implement this for the MIND prediction: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(Block):
    def predict(
        self, predictions, targets=None, training=True, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")


class RemovePad3D(Block):
    """
    Flatten the sequence of labels and filter out non-targets positions

    Parameters
    ----------
        padding_idx: int
            The padding index value.
            Defaults to 0.

    Returns
    -------
        targets: tf.Tensor
            The flattened vector of true targets positions
        flatten_predictions: tf.Tensor
            If the predicions are 3-D vectors (sequential task),
            flatten the predictions vectors to keep only the ones related to target positions.
    """

    def __init__(self, padding_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = padding_idx

    def call_targets(self, predictions, targets, training=True, **kwargs) -> tf.Tensor:
        targets = tf.reshape(targets, (-1,))
        non_pad_mask = targets != self.padding_idx
        targets = tf.boolean_mask(targets, non_pad_mask)

        if len(tuple(predictions.get_shape())) == 3:
            predictions = tf.reshape(predictions, (-1, predictions.shape[-1]))
            flatten_predictions = tf.boolean_mask(
                predictions, tf.broadcast_to(tf.expand_dims(non_pad_mask, 1), tf.shape(predictions))
            )
            return targets, flatten_predictions
        return targets


class MaskingHead(Block):
    """
    The masking class to transform targets based on the
    boolean masking schema stored in the model's context

    Parameters
    ----------
        padding_idx: int
            The padding index value.
            Defaults to 0.

    Returns
    -------
        targets: tf.Tensor
            Tensor of masked labels.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0

    def call_targets(
        self, predictions: tf.Tensor, targets: tf.Tensor, training: bool = True, **kwargs
    ) -> tf.Tensor:
        targets = self.context[Tag.ITEM_ID]
        mask = self.context.get_mask()
        targets = tf.where(mask, targets, self.padding_idx)
        return targets


def NextItemPredictionTask(
    schema,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
    ),
    metrics=ranking_metrics(top_ks=[10, 20], labels_onehot=True),
    weight_tying: bool = True,
    masking: bool = True,
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    softmax_temperature: float = 1,
    normalize: bool = True,
    sampled_softmax: bool = False,
    num_sampled: int = 100,
) -> MultiClassClassificationTask:
    """
    Function to create the NextItemPrediction task with the right parameters.

    Parameters
    ----------
        schema:
        loss: tf.keras.losses.Loss
            Loss function.
            Defaults to `tf.keras.losses.SparseCategoricalCrossentropy()`.
        metrics: Sequence[MetricOrMetricClass]
            List of top-k ranking metrics.
            Defaults to ranking_metrics(top_ks=[10, 20], labels_onehot=True).
        weight_tying: bool
            The item_id embedding weights are shared with the prediction network layer.
            Defaults to True
        masking: bool
            Whether masking is used to transform inputs and targets or not
            Defaults to True
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
            Parameter used to reduce the model overconfidence, so that softmax(logits / T).
            Defaults to 1.
        normalize: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to True.
        sampled_softmax: bool
            Compute the logits scores over all items of the catalog or
            generate a subset of candidates
            When set to True, loss should be set to `tf.nn.softmax_cross_entropy_with_logits`
            and metrics to `ranking_metrics(top_ks=..., labels_onehot=False)`
            Defaults to False
        num_sampled: int
            When sampled_softmax is enabled, specify the number of
            negative candidates to generate for each batch
            Defaults to 100

    Returns
    -------
        PredictionTask
            The next item prediction task
    """
    if normalize:
        prediction_call = L2Norm()

    if masking:
        prediction_call = prediction_call.connect(MaskingHead())
        prediction_call = prediction_call.connect(RemovePad3D())

    if sampled_softmax:
        prediction_call = prediction_call.connect(
            SampledSoftmax(schema, num_sampled=num_sampled, weight_tying=weight_tying)
        )

    elif weight_tying:
        prediction_call = prediction_call.connect(ItemSoftmaxWeightTying(schema))

    else:
        prediction_call = prediction_call.connect(Softmax(schema))

    if softmax_temperature != 1:
        prediction_call = prediction_call.connect(SoftmaxTemperature(softmax_temperature))

    if extra_pre_call is not None:
        prediction_call = prediction_call.connect(extra_pre_call)

    return MultiClassClassificationTask(
        target_name,
        task_name,
        task_block,
        loss=loss,
        metrics=metrics,
        pre=prediction_call,
    )


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
    prediction_call = InBatchNegativeSampling()

    if normalize:
        prediction_call = L2Norm().connect(prediction_call)

    if softmax_temperature != 1:
        prediction_call = prediction_call.connect(SoftmaxTemperature(softmax_temperature))

    if extra_pre_call is not None:
        prediction_call = prediction_call.connect(extra_pre_call)

    return MultiClassClassificationTask(
        target_name,
        task_name,
        task_block,
        loss=loss,
        metrics=metrics,
        pre=prediction_call,
    )


def YoutubeDNNRetrieval(
    schema,
    aggregation: str = "concat",
    top_layer: Optional[Block] = MLPBlock([64]),
    weight_tying: bool = True,
    sampled_softmax: Optional[bool] = True,
    num_sampled: int = 100,
    loss=tf.nn.softmax_cross_entropy_with_logits,
    metrics=ranking_metrics(top_ks=[10, 20], labels_onehot=False),
    normalize: bool = True,
    extra_pre_call: Optional[Block] = None,
    task_block: Optional[Layer] = None,
    softmax_temperature: float = 1,
    seq_aggregator: Block = SequenceAggregator(SequenceAggregation.MEAN),
):

    """
    Build the Youtube-DNN retrieval model.
    More details of the model can be found at
    [Covington et al., 2016](https://dl.acm.org/doi/10.1145/2959100.2959190Covington)
    """

    inputs = InputBlock(
        schema,
        aggregation=aggregation,
        seq=False,
        masking="clm",
        split_sparse=True,
        seq_aggregator=seq_aggregator,
    )

    task = NextItemPredictionTask(
        schema=schema,
        loss=loss,
        metrics=metrics,
        masking=True,
        weight_tying=weight_tying,
        sampled_softmax=sampled_softmax,
        extra_pre_call=extra_pre_call,
        task_block=task_block,
        softmax_temperature=softmax_temperature,
        normalize=normalize,
        num_sampled=num_sampled,
    )

    return inputs.connect(top_layer, task)
