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
from typing import List, Optional, Sequence, Union, TypedDict, Text

import tensorflow as tf
from tensorflow.keras.layers import Layer
from merlin.models.tf.prediction_tasks.base import PredictionTask

from merlin.models.tf.blocks.sampling.base import ItemSampler, Items
from merlin.schema import Schema, Tags

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    EmbeddingWithMetadata,
    PredictionOutput, TaskWithOutputs, MetricOrMetrics,
)
from merlin.models.tf.prediction_tasks.classification import CategoricalPrediction
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT

LOG = logging.getLogger("merlin_models")

"""
# Retrieval case
model = TwoTowerModel(schema, MLPBlock([512, 128]))
model.compile(prediction_task=ContrastiveLearningTask(schema, "in-batch"))
model.save(save_pre_loss=False)

new_model = model.to_item_recommender(train)
model.compile(prediction_task=ContrastiveLearningTask(schema, "in-batch", "cross-batch"))

# Ranking case
pred_task = NextItemPredictionTask(schema)
model = DLRMModel(schema, 128, MLPBlock([512, 128]), prediction_tasks=pred_task)
model.compile(prediction_task=ContrastiveLearningTask(schema, "popularity-based"))
"""


@dataclass
class ContrastiveRepresentation:
    query: Items
    positive: Items
    negative: Items

    @property
    def shape(self) -> "ContrastiveRepresentation":
        return ContrastiveRepresentation(self.query.shape, self.positive.shape, self.negative.shape)


class ContrastiveLearningTask(PredictionTask):
    DEFAULT_LOSS = "categorical_crossentropy"
    DEFAULT_METRICS: MetricOrMetrics = (tf.keras.metrics.Accuracy,)

    def __init__(
            self,
            schema: Schema,
            *samplers: ItemSampler,
            prediction_block: Optional[CategoricalPrediction] = None,
            item_metadata_schema: Optional[Schema] = None,
            item_id_tag: Tags = Tags.ITEM_ID,
            query_id_tag: Tags = Tags.USER_ID,
            downscore_false_negatives: bool = True,
            false_negative_score: float = MIN_FLOAT,
            target_name: Optional[str] = None,
            task_name: Optional[str] = None,
            pre: Optional[Block] = None,
            post: Optional[Block] = None,
            task_block: Optional[Layer] = None,
            name: Optional[Text] = None,
            **kwargs,
    ):
        self.samplers = list(samplers)
        self.schema = schema
        self.item_metadata_schema = item_metadata_schema
        self.item_id_feature_name = schema.select_by_tag(item_id_tag).first.name
        self.query_id_feature_name = schema.select_by_tag(query_id_tag).first.name
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.prediction_block = prediction_block

        super().__init__(target_name=target_name, task_name=task_name, pre=pre, post=post, task_block=task_block,
                         name=name, **kwargs)

    def call(
            self,
            inputs: Union[TabularData, tf.Tensor],
            training=False,
            testing=False,
    ) -> tf.Tensor:
        if not (training or testing):
            if isinstance(inputs, dict):
                scores = tf.reduce_sum(
                    tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
                )
                return scores
            else:
                return self.prediction_block(inputs, training=training, testing=testing)

        representation = self.create_representation(inputs, training, testing)
        predictions = self.process_representation(representation, training, testing)

        return predictions

    def create_representation(
            self,
            inputs: Union[TabularData, tf.Tensor],
            training=False,
            testing=False,
    ) -> ContrastiveRepresentation:
        queries: Items = self.queries(inputs, training, testing)
        positive_items: Items = self.positive_items(inputs, training, testing)
        negative_items: Items = self.sample_negatives(inputs, positive_items, training, testing)

        representation = ContrastiveRepresentation(queries, positive_items, negative_items)

        if not representation.positive.has_embedding:
            if not self.prediction_block:
                raise ValueError("No prediction block provided for task")
            representation.positive = representation.positive.with_embedding(
                self.prediction_block.embedding_lookup(self.positive.ids)
            )
        if not representation.negative.has_embedding:
            representation.negative = representation.negative.with_embedding(
                self.prediction_block.embedding_lookup(self.negative.ids)
            )

        return representation

    def queries(
            self,
            inputs: Union[TabularData, tf.Tensor],
            training=False,
            testing=False,
    ) -> Items:
        del training, testing

        if isinstance(inputs, tf.Tensor):
            query = inputs
            query_id = self.context[self.query_id_feature_name]
        elif isinstance(inputs, dict) and "query" in inputs:
            query = inputs["query"]
            if "query_id" in inputs:
                query_id = inputs["query_id"]
            else:
                query_id = self.context[self.query_id_feature_name]
        else:
            raise ValueError(f"Task results does not contain query. Got: {inputs}")

        return Items(query_id).with_embedding(query)

    def positive_items(
            self,
            inputs: Union[TabularData, tf.Tensor],
            training=False,
            testing=False,
    ) -> Items:
        del training, testing

        if isinstance(inputs, dict) and "item_id" in inputs:
            item_ids = inputs["item_id"]
        else:
            item_ids = self.context[self.item_id_feature_name]
        item_metadata: TabularData = {}
        item_features = self.get_features_from_context()
        if item_features:
            item_metadata.update(item_features)

        output = Items(item_ids, item_metadata)
        if isinstance(inputs, dict) and "item" in inputs:
            output = output.with_embedding(inputs["item"])

        return output

    def sample_negatives(
            self,
            inputs: Union[TabularData, tf.Tensor],
            positive_items: Items,
            training=False,
            testing=False,
    ) -> Items:
        negative_items: List[Items] = []

        # Adds items from the current batch into samplers and sample a number of negatives
        for sampler in self.samplers:
            sampling_kwargs = {"training": training}
            extra_args = {"testing": testing, "inputs": inputs}
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
            training=False,
            testing=False,
    ) -> tf.Tensor:
        predictions = {}

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

        return predictions

    def pre_loss(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
        predictions = outputs.predictions
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

        return outputs.copy_with_updates(targets=targets)

    def get_features_from_context(self) -> TabularData:
        if not self.item_metadata_schema:
            return {}

        columns = set(self.item_metadata_schema.column_names)

        return {key: self.context[key] for key in columns}

    def add_features_to_context(self, shapes):
        features = self.item_metadata_schema.column_names if self.item_metadata_schema else []
        if self.prediction_block:
            features.append(self.item_id_feature_name)

        return list(set(features))

    def add_sampler(self, sampler: ItemSampler):
        self.samplers.append(sampler)

        return self
