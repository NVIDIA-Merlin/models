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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, TypedDict

import tensorflow as tf
from tensorflow.keras.layers import Layer
from merlin.models.tf.prediction_tasks.base import PredictionTask

from merlin.models.tf.blocks.sampling.base import ItemSampler, Items
from merlin.schema import Schema, Tags
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    EmbeddingWithMetadata,
    PredictionOutput, TaskWithOutputs,
)
from merlin.models.tf.prediction_blocks.base import PreLossBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.models.tf.models.base import Model

LOG = logging.getLogger("merlin_models")

"""
# Retrieval case
model = TwoTowerModel(schema, MLPBlock([512, 128]))
model.compile(pre_loss=ContrastiveLearning(schema, "in-batch"))
model.save(save_pre_loss=False)

new_model = model.to_item_recommender(train)
model.compile(pre_loss=ContrastiveLearning(schema, "in-batch", "cross-batch"))

# Ranking case
pred_task = NextItemPredictionTask(schema)
model = DLRMModel(schema, 128, MLPBlock([512, 128]), prediction_tasks=pred_task)
model.compile(pre_loss=ContrastiveLearning(schema, "popularity-based"))
"""


@dataclass
class ContrastiveRepresentation:
    query: Items
    positive: Items
    negative: Items

    @property
    def shape(self) -> "ContrastiveRepresentation":
        return ContrastiveRepresentation(self.query.shape, self.positive.shape, self.negative.shape)


class ContrastiveRepresentationTransform(Layer):
    def call(
            self,
            inputs: ContrastiveRepresentation,
            model: Model,
            task: PredictionTask
    ) -> ContrastiveRepresentation:
        # TODO: Optimize this by turning it into a single call to the embedding layer

        if not inputs.positive.has_embedding:
            inputs.positive = inputs.positive.with_embedding(task.block.embedding_lookup(self.positive.ids))
        if not inputs.negative.has_embedding:
            inputs.negative = inputs.negative.with_embedding(task.block.embedding_lookup(self.negative.ids))

        return inputs


class ContrastiveLearning(PreLossBlock):
    def __init__(
            self,
            schema: Schema,
            *samplers: ItemSampler,
            item_metadata_schema: Optional[Schema] = None,
            item_id_tag: Tags = Tags.ITEM_ID,
            query_id_tag: Tags = Tags.USER_ID,
            downscore_false_negatives: bool = True,
            false_negative_score: float = MIN_FLOAT,
            post: Optional[ContrastiveRepresentationTransform] = None,
            **kwargs,
    ):
        self.samplers = samplers
        self.schema = schema
        self.item_metadata_schema = item_metadata_schema
        self.post = post or ContrastiveRepresentationTransform()
        self.item_id_feature_name = schema.select_by_tag(item_id_tag).first.name
        self.query_id_feature_name = schema.select_by_tag(query_id_tag).first.name
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score

        super().__init__(**kwargs)

    def call(
            self,
            features: TabularData,
            task_results: TaskWithOutputs,
            model: Model,
            training=False,
            testing=False,
    ) -> TaskWithOutputs:
        representation = self.create_representation(
            features, task_results, model, training, testing
        )
        task_outputs = self.process_representation(
            representation, task_results.task, training, testing
        )

        return task_outputs

    def create_representation(
            self,
            features: TabularData,
            task_results: TaskWithOutputs,
            model: Model,
            training=False,
            testing=False,
    ) -> ContrastiveRepresentation:
        queries: Items = self.queries(
            features, task_results, model, training, testing
        )
        positive_items: Items = self.positive_items(
            features, task_results, model, training, testing
        )
        negative_items: Items = self.sample_negatives(
            task_results, positive_items, model, training, testing
        )

        representation = ContrastiveRepresentation(queries, positive_items, negative_items)
        representation = self.post(representation, model, task_results.task)

        return representation

    def queries(
            self,
            features: TabularData,
            task_results: TaskWithOutputs,
            model: Model,
            training=False,
            testing=False,
    ) -> Items:
        del model, training, testing

        if isinstance(task_results.predictions, tf.Tensor):
            query = task_results.predictions
            query_id = features[self.query_id_feature_name]
        elif isinstance(task_results.predictions, dict) and "query" in task_results.predictions:
            query = task_results.predictions["query"]
            if "query_id" in task_results.predictions:
                query_id = task_results.predictions["query_id"]
            else:
                query_id = features[self.query_id_feature_name]
        else:
            raise ValueError(f"Task results does not contain query. Got: {task_results}")

        return Items(query_id).with_embedding(query)

    def positive_items(
            self,
            features: TabularData,
            task_results: TaskWithOutputs,
            model: Model,
            training=False,
            testing=False,
    ) -> Items:
        del model, training, testing

        item_ids = features[self.item_id_feature_name]
        item_metadata: TabularData = {}
        item_features = self.filter_features(features)
        if item_features:
            item_metadata.update(item_features)

        output = Items(item_ids, item_metadata)
        if isinstance(task_results.predictions, dict) and "item" in task_results.predictions:
            output = output.with_embedding(task_results.predictions["item"])

        return output

    def sample_negatives(
            self,
            task_results: TaskWithOutputs,
            positive_items: Items,
            model: Model,
            training=False,
            testing=False,
    ) -> Items:
        negative_items: List[Items] = []

        # Adds items from the current batch into samplers and sample a number of negatives
        for sampler in self.samplers:
            sampling_kwargs = {"training": training}
            extra_args = {"testing": testing, "model": model, "task_results": task_results}
            for name, val in extra_args.items():
                if name in sampler._call_fn_args:
                    sampling_kwargs[name] = val

            sampler_items: Items = sampler(positive_items, **sampling_kwargs)

            if tf.shape(sampler_items.ids)[0] > 0:
                negative_items.append(sampler_items)
            else:
                LOG.warn(
                    f"The sampler {type(sampler).__name__} returned no samples for this batch."
                )

        if len(negative_items) == 0:
            raise Exception(f"No negative items where sampled from samplers {self.samplers}")

        negatives = sum(negative_items) if len(negative_items) > 1 else negative_items[0]

        return negatives

    def process_representation(
            self,
            representation: ContrastiveRepresentation,
            task: PredictionTask,
            training=False,
            testing=False,
    ) -> TaskWithOutputs:
        predictions, targets = {}, {}

        positive_scores = tf.reduce_sum(
            tf.multiply(representation.query.embedding(), representation.positive.embedding()),
            keepdims=True,
            axis=-1,
        )

        if training or testing:
            negative_scores = tf.linalg.matmul(
                representation.query.embedding(), representation.negative.embedding(), transpose_b=True
            )

            if self.downscore_false_negatives:
                negative_scores, valid_negatives_mask = rescore_false_negatives(
                    representation.positive.ids,
                    representation.negative.ids,
                    negative_scores,
                    self.false_negative_score
                )

            predictions = tf.concat([positive_scores, negative_scores], axis=-1)

            # To ensure that the output is always fp32, avoiding numerical
            # instabilities with mixed_float16 policy
            predictions = tf.cast(predictions, tf.float32)

        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"

        targets = tf.concat(
            [
                tf.ones([tf.shape(predictions)[0], 1], dtype=predictions.dtype),
                tf.zeros(
                    [tf.shape(predictions)[0], tf.shape(predictions)[1] - 1],
                    dtype=predictions.dtype,
                ),
            ],
            axis=1,
        )

        return TaskWithOutputs(task, predictions, targets)

    def filter_features(self, features: TabularData) -> TabularData:
        if self.item_metadata_schema:
            return {}
        columns = set(self.item_metadata_schema.column_names)

        return {key: val for key, val in features.items() if key in columns}


# @tf.keras.utils.register_keras_serializable(package="merlin_models")
# class ContrastiveScorer(Block):
#     def __init__(
#             self,
#             schema: Schema,
#             downscore_false_negatives: bool = True,
#             false_negative_score: float = MIN_FLOAT,
#             **kwargs,
#     ):
#         super().__init__(schema=schema, **kwargs)
#
#         self.downscore_false_negatives = downscore_false_negatives
#         self.false_negative_score = false_negative_score
#         self.item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
#
#     def call(
#             self,
#             inputs: ContrastiveInputs,
#             training: bool = False,
#             testing: bool = False,
#             **kwargs,
#     ) -> Union[tf.Tensor, TabularData]:
#         """Based on the user/query embedding (inputs[self.query_name]), uses dot product to score
#             the positive item (inputs["item"]).
#             For the sampled-softmax mode, logits are computed by multiplying the query vector
#             and the item embeddings matrix (self.context.get_embedding(self.item_column)))
#         Parameters
#         ----------
#         inputs : Union[tf.Tensor, TabularData]
#             Dict with the query and item embeddings (e.g. `{"query": <emb>}, "item": <emb>}`),
#             where embeddings are 2D tensors (batch size, embedding size)
#         training : bool, optional
#             Flag that indicates whether in training mode, by default True
#         Returns
#         -------
#         tf.Tensor
#             2D Tensor with the scores for the positive items,
#             If `training=True`, return the original inputs
#         """
#         if training or testing:
#             return inputs
#
#         # How to get embedding-table here?
#         if callable(inputs["item"]):
#             return tf.matmul(inputs, tf.transpose(embedding_table))
#
#         if self.sampled_softmax_mode:
#             return self._get_logits_for_sampled_softmax(inputs)
#
#         self._check_input_from_two_tower(inputs)
#         positive_scores = tf.reduce_sum(
#             tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
#         )
#         return positive_scores
#
#     # @tf.function
#     def call_outputs(
#             self, outputs: PredictionOutput, training=True, testing=False, **kwargs
#     ) -> "PredictionOutput":
#         """Based on the user/query embedding (inputs[self.query_name]), uses dot product to score
#             the positive item and also sampled negative items (during training).
#         Parameters
#         ----------
#         inputs : TabularData
#             Dict with the query and item embeddings (e.g. `{"query": <emb>}, "item": <emb>}`),
#             where embeddings are 2D tensors (batch size, embedding size)
#         training : bool, optional
#             Flag that indicates whether in training mode, by default True
#         Returns
#         -------
#         [tf.Tensor,tf.Tensor]
#             all_scores: 2D Tensor with the scores for the positive items and, if `training=True`,
#             for the negative sampled items too.
#             Return tensor is 2D (batch size, 1 + #negatives)
#         """
#         valid_negatives_mask = None
#         predictions: DualEncoderOutputs = outputs.predictions
#         targets = outputs.targets
#
#         if self.sampled_softmax_mode or isinstance(targets, tf.Tensor):
#             positive_item_ids = targets
#         else:
#             positive_item_ids = predictions["item_id"]
#
#         positive_scores = tf.reduce_sum(
#             tf.multiply(predictions["query"], predictions["item"]),
#             keepdims=True,
#             axis=-1,
#         )
#
#         if "negative_item" not in predictions:
#             return PredictionOutput(positive_scores, targets)
#
#         if training or testing:
#             if self.sampled_softmax_mode:
#                 predictions = self._prepare_query_item_vectors_for_sampled_softmax(
#                     predictions, targets
#                 )
#
#             positive_scores = tf.reduce_sum(
#                 tf.multiply(predictions["query"], predictions["item"]),
#                 keepdims=True,
#                 axis=-1,
#             )
#
#             negative_scores = tf.linalg.matmul(
#                 predictions["query"], predictions["negative_item"], transpose_b=True
#             )
#
#             if self.downscore_false_negatives and "negative_item_id" in predictions:
#                 negative_scores, valid_negatives_mask = rescore_false_negatives(
#                     positive_item_ids,
#                     predictions["negative_item_id"],
#                     negative_scores,
#                     self.false_negatives_score
#                 )
#
#             predictions = tf.concat([positive_scores, negative_scores], axis=-1)
#
#             # To ensure that the output is always fp32, avoiding numerical
#             # instabilities with mixed_float16 policy
#             predictions = tf.cast(predictions, tf.float32)
#
#         assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
#         # prepare targets for computing the loss and metrics
#         if self.sampled_softmax_mode and not training:
#             # Converts target ids to one-hot representation
#             num_classes = tf.shape(predictions)[-1]
#             targets_one_hot = tf.one_hot(tf.reshape(targets, (-1,)), num_classes)
#             return PredictionOutput(
#                 predictions,
#                 targets_one_hot,
#                 positive_item_ids=positive_item_ids,
#                 valid_negatives_mask=valid_negatives_mask,
#             )
#         else:
#             # Positives in the first column and negatives in the subsequent columns
#             targets = tf.concat(
#                 [
#                     tf.ones([tf.shape(predictions)[0], 1], dtype=predictions.dtype),
#                     tf.zeros(
#                         [tf.shape(predictions)[0], tf.shape(predictions)[1] - 1],
#                         dtype=predictions.dtype,
#                     ),
#                 ],
#                 axis=1,
#             )
#             return PredictionOutput(
#                 predictions,
#                 targets,
#                 positive_item_ids=positive_item_ids,
#                 valid_negatives_mask=valid_negatives_mask,
#             )
#
#     def _get_logits_for_sampled_softmax(self, inputs):
#         if not isinstance(inputs, tf.Tensor):
#             raise ValueError(
#                 f"Inputs to the Sampled Softmax block should be tensors, got {type(inputs)}"
#             )
#         embedding_table = self.context.get_embedding(self.item_id_feature_name)
#         all_scores = tf.matmul(inputs, tf.transpose(embedding_table))
#         return all_scores
#
#     def _prepare_query_item_vectors_for_sampled_softmax(
#             self, predictions: tf.Tensor, targets: tf.Tensor
#     ):
#         # extract positive items embeddings
#         if not isinstance(predictions, tf.Tensor):
#             raise ValueError(
#                 f"Inputs to the Sampled Softmax block should be tensors, got {type(predictions)}"
#             )
#         embedding_table = self.context.get_embedding(self.item_id_feature_name)
#         batch_items_embeddings = embedding_ops.embedding_lookup(embedding_table, targets)
#         predictions = {"query": predictions, "item": batch_items_embeddings}
#         return predictions
#
#     def get_config(self):
#         config = super().get_config()
#         config = maybe_serialize_keras_objects(self, config, ["samplers"])
#         config["downscore_false_negatives"] = self.downscore_false_negatives
#         config["false_negatives_score"] = self.false_negatives_score
#         config["item_id_feature_name"] = self.item_id_feature_name
#
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         config = maybe_deserialize_keras_objects(config, ["samplers"])
#
#         return super().from_config(config)


def _list_to_tensor(input_list: List[tf.Tensor]) -> tf.Tensor:
    output: tf.Tensor

    if len(input_list) == 1:
        output = input_list[0]
    else:
        output = tf.concat(input_list, axis=0)

    return output
