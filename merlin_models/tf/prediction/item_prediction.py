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
import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.base import Layer

from merlin_models.tf.block.transformations import L2Norm
from merlin_models.tf.core import Block, ItemSampler, ItemSamplerData
from merlin_standard_lib import Schema, Tag

from .classification import MultiClassClassificationTask
from .ranking_metric import ranking_metrics

LOG = logging.getLogger("merlin_models")

MIN_FLOAT = np.finfo(np.float32).min / 100.0


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


# TODO: Implement this for the MIND prediction: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(Block):
    def predict(
        self, predictions, targets=None, training=True, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")


@Block.registry.register_with_multiple_names("item_retrieval_scorer")
class ItemRetrievalScorer(Block):
    def __init__(
        self,
        samplers: List[ItemSampler] = [],
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value: int = MIN_FLOAT,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value

        self.samplers = samplers
        if not isinstance(self.samplers, list):
            self.samplers = [self.samplers]

        self.set_required_features()

        self.max_num_samples = 0
        for sampler in self.samplers:
            self.max_num_samples += sampler.max_num_samples

    def set_required_features(self):
        required_features = set()
        if self.downscore_false_negatives:
            required_features.add(str(Tag.ITEM_ID))

        required_features.update(
            [feature for sampler in self.samplers for feature in sampler.required_features]
        )

        self._required_features = list(required_features)

    def add_features_to_context(self, feature_shapes) -> List[str]:
        return self._required_features

    def _check_input_from_two_tower(self, inputs):
        assert set(inputs.keys()) == set(["query", "item"])

    def _check_required_context_item_features_are_present(self):
        not_found = list(
            [
                feat_name
                for feat_name in self._required_features
                if getattr(self, "_context", None) is None
                or feat_name not in self.context.named_variables
            ]
        )

        if len(not_found) > 0:
            raise ValueError(
                f"The following required context features should be available "
                f"for the samplers, but were not found: {not_found}"
            )

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        self._check_input_from_two_tower(inputs)
        self._check_required_context_item_features_are_present()

        positive_scores = tf.reduce_sum(
            tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
        )

        if not training:
            return positive_scores
        else:
            assert (
                len(self.samplers) > 0
            ), "At least one sampler is required by ItemRetrievalScorer for negative sampling"

            batch_items_embeddings = inputs["item"]
            batch_items_metadata = self.get_batch_items_metadata()

            neg_items_embeddings_list = []
            neg_items_ids_list = []

            # Adds items from the current batch into samplers and sample a number of negatives
            for sampler in self.samplers:
                input_data = ItemSamplerData(batch_items_embeddings, batch_items_metadata)
                neg_items = sampler(input_data.__dict__)

                if tf.shape(neg_items.items_embeddings)[0] > 0:
                    # Accumulates sampled negative items from all samplers
                    neg_items_embeddings_list.append(neg_items.items_embeddings)
                    if self.downscore_false_negatives:
                        neg_items_ids_list.append(neg_items.items_metadata[str(Tag.ITEM_ID)])
                else:
                    LOG.warn(
                        f"The sampler {type(sampler).__name__} returned no samples for this batch."
                    )

            if len(neg_items_embeddings_list) == 0:
                raise Exception(f"No negative items where sampled from samplers {self.samplers}")
            elif len(neg_items_embeddings_list) == 1:
                neg_items_embeddings = neg_items_embeddings_list[0]
            else:
                neg_items_embeddings = tf.concat(neg_items_embeddings_list, axis=0)

            negative_scores = tf.linalg.matmul(
                inputs["query"], neg_items_embeddings, transpose_b=True
            )

            if self.downscore_false_negatives:
                if len(neg_items_ids_list) == 1:
                    neg_items_ids = neg_items_ids_list[0]
                else:
                    neg_items_ids = tf.concat(neg_items_ids_list, axis=0)

                positive_item_ids = self.context[Tag.ITEM_ID]
                negative_scores = self.rescore_false_negatives(
                    positive_item_ids, neg_items_ids, negative_scores
                )

            all_scores = tf.concat([positive_scores, negative_scores], axis=-1)
            return all_scores

    def call_targets(self, predictions, targets, **kwargs) -> tf.Tensor:
        # Positives in the first column and negatives in the subsequent columns
        targets = tf.concat(
            [
                tf.ones([tf.shape(predictions)[0], 1]),
                tf.zeros([tf.shape(predictions)[0], tf.shape(predictions)[1] - 1]),
            ],
            axis=1,
        )
        return targets

    def get_batch_items_metadata(self):
        result = {feat_name: self.context[feat_name] for feat_name in self._required_features}
        return result

    def rescore_false_negatives(self, positive_item_ids, neg_samples_item_ids, negative_scores):
        false_negatives_mask = tf.equal(tf.expand_dims(positive_item_ids, -1), neg_samples_item_ids)
        # Setting a very small value for false negatives (accidental hits) so that it has
        # negligicle effect on the loss functions
        negative_scores = tf.where(
            false_negatives_mask,
            tf.ones_like(negative_scores) * self.false_negatives_score,
            negative_scores,
        )

        return negative_scores

    # def compute_output_shape(self, input_shape):
    #    return (100, 401)


def ItemRetrievalTask(
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    ),
    samplers: List[ItemSampler] = [],
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
        samplers: List[ItemSampler]
            List of samplers for negative sampling
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

    if samplers is None or (isinstance(samplers, list) and len(samplers) == 0):
        # samplers = [InBatchSampler(batch_size=batch_size)]
        raise ValueError(
            "You must provide at least one sampler "
            + "(e.g. `samplers = [InBatchSampler(), CachedBatchesSampler()]`) for ItemRetrievalTask"
        )

    prediction_call = ItemRetrievalScorer(
        samplers=samplers,
    )

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
