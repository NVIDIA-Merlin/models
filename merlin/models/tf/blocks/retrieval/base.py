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
from typing import List, Sequence, Union

import tensorflow as tf

from merlin.schema import Schema, Tags
from tensorflow.python.ops import embedding_ops

from ....utils.constants import MIN_FLOAT
from ...core import Block, EmbeddingWithMetadata, ModelBlock
from ...typing import TabularData
from ...utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects
from ..sampling.base import ItemSampler

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TowerBlock(ModelBlock):
    pass


class RetrievalMixin:
    def query_block(self) -> TowerBlock:
        raise NotImplementedError()

    def item_block(self) -> TowerBlock:
        raise NotImplementedError()


@Block.registry.register_with_multiple_names("item_retrieval_scorer")
class ItemRetrievalScorer(Block):
    """Block for ItemRetrieval, which expects query/user and item embeddings as input and
    uses dot product to score the positive item (inputs["item"]) and also sampled negative
    items (during training).

    Parameters
    ----------
    samplers : List[ItemSampler], optional
        List of item samplers that provide negative samples when `training=True`
    sampling_downscore_false_negatives : bool, optional
        Identify false negatives (sampled item ids equal to the positive item and downscore them
        to the `sampling_downscore_false_negatives_value`), by default True
    sampling_downscore_false_negatives_value : int, optional
        Value to be used to downscore false negatives when
        `sampling_downscore_false_negatives=True`, by default `np.finfo(np.float32).min / 100.0`
    item_id_feature_name: str
        Name of the column containing the item ids
        Defaults to `item_id`
    """

    def __init__(
        self,
        samplers: Sequence[ItemSampler] = (),
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value: int = MIN_FLOAT,
        item_id_feature_name: str = "item_id",
        **kwargs,
    ):
        if not isinstance(samplers, (list, tuple)):
            samplers = (samplers,)

        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value
        self.item_id_feature_name = item_id_feature_name
        self.samplers = samplers

        self.set_required_features()

    def build(self, input_shapes):
        if isinstance(input_shapes, dict):
            query_shape = input_shapes["query"]
            self.context.add_variable(
                tf.Variable(
                    initial_value=tf.zeros([1, query_shape[-1]], dtype=tf.float32),
                    name="query",
                    trainable=False,
                    validate_shape=False,
                    shape=tf.TensorShape([None, query_shape[-1]]),
                )
            )
        super().build(input_shapes)

    def set_required_features(self):
        required_features = set()
        if self.downscore_false_negatives:
            required_features.add(self.item_id_feature_name)

        required_features.update(
            [feature for sampler in self.samplers for feature in sampler.required_features]
        )

        self._required_features = list(required_features)

    def add_features_to_context(self, feature_shapes) -> List[str]:
        return self._required_features

    def _check_input_from_two_tower(self, inputs):
        assert set(inputs.keys()) == {"query", "item"}

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

    def call(
        self, inputs: Union[tf.Tensor, TabularData], training: bool = False, **kwargs
    ) -> Union[tf.Tensor, TabularData]:
        """Based on the user/query embedding (inputs["query"]), uses dot product to score
            the positive item (inputs["item"] or  self.context.get_embedding(self.item_column))

        Parameters
        ----------
        inputs : Union[tf.Tensor, TabularData]
            Dict with the query and item embeddings (e.g. `{"query": <emb>}, "item": <emb>}`),
            where embeddings are 2D tensors (batch size, embedding size)
        training : bool, optional
            Flag that indicates whether in training mode, by default True

        Returns
        -------
        tf.Tensor
            2D Tensor with the scores for the positive items,
            If `training=True`, return the original inputs
        """
        if isinstance(inputs, dict):
            self.context["query"].assign(inputs["query"])

        if training:
            return inputs

        if isinstance(inputs, tf.Tensor):
            embedding_table = self.context.get_embedding(self.item_id_feature_name)
            all_scores = tf.matmul(inputs, tf.transpose(embedding_table))
            return all_scores

        self._check_input_from_two_tower(inputs)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
        )
        return positive_scores

    @tf.function
    def call_targets(
        self,
        predictions: Union[tf.Tensor, TabularData],
        targets: tf.Tensor,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Based on the user/query embedding (inputs["query"]), uses dot product to score
            the positive item and also sampled negative items (during training).

        Parameters
        ----------
        predictions : Union[TabularData, tf.Tensor]
            Dict with the query and item embeddings (e.g. `{"query": <emb>}, "item": <emb>}`),
            where embeddings are 2D tensors (batch size, embedding size)
        training : bool, optional
            Flag that indicates whether in training mode, by default True

        Returns
        -------
        [tf.Tensor,tf.Tensor]
            predictions: 2D Tensor with the scores for the positive items and, if `training=True`,
            for the negative sampled items too.
            Returned tensor is 2D (batch size, 1 + #negatives)
        """
        self._check_required_context_item_features_are_present()

        if training:
            assert (
                len(self.samplers) > 0
            ), "At least one sampler is required by ItemRetrievalScorer for negative sampling"

            if isinstance(predictions, dict):
                batch_items_embeddings = predictions["item"]
            else:
                embedding_table = self.context.get_embedding(self.item_id_feature_name)
                batch_items_embeddings = embedding_ops.embedding_lookup(embedding_table, targets)
                predictions = {"query": predictions, "item": batch_items_embeddings}
            batch_items_metadata = self.get_batch_items_metadata()

            positive_scores = tf.reduce_sum(
                tf.multiply(predictions["query"], predictions["item"]), keepdims=True, axis=-1
            )

            neg_items_embeddings_list = []
            neg_items_ids_list = []

            # Adds items from the current batch into samplers and sample a number of negatives
            for sampler in self.samplers:
                input_data = EmbeddingWithMetadata(batch_items_embeddings, batch_items_metadata)
                if "item_weights" in sampler._call_fn_args:
                    neg_items = sampler(input_data.__dict__, item_weights=embedding_table)
                else:
                    neg_items = sampler(input_data.__dict__)

                if tf.shape(neg_items.embeddings)[0] > 0:
                    # Accumulates sampled negative items from all samplers
                    neg_items_embeddings_list.append(neg_items.embeddings)
                    if self.downscore_false_negatives:
                        neg_items_ids_list.append(neg_items.metadata[self.item_id_feature_name])
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
                predictions["query"], neg_items_embeddings, transpose_b=True
            )

            if self.downscore_false_negatives:
                if isinstance(targets, tf.Tensor):
                    positive_item_ids = targets
                else:
                    positive_item_ids = self.context[self.item_id_feature_name]

                if len(neg_items_ids_list) == 1:
                    neg_items_ids = neg_items_ids_list[0]
                else:
                    neg_items_ids = tf.concat(neg_items_ids_list, axis=0)

                negative_scores = self.rescore_false_negatives(
                    positive_item_ids, neg_items_ids, negative_scores
                )

            predictions = tf.concat([positive_scores, negative_scores], axis=-1)

        # Positives in the first column and negatives in the subsequent columns
        targets = tf.concat(
            [
                tf.ones([tf.shape(predictions)[0], 1]),
                tf.zeros([tf.shape(predictions)[0], tf.shape(predictions)[1] - 1]),
            ],
            axis=1,
        )
        return targets, predictions

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

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, ["samplers"])
        config["downscore_false_negatives"] = self.downscore_false_negatives
        config["false_negatives_score"] = self.false_negatives_score
        config["item_id_feature_name"] = self.item_id_feature_name

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["samplers"])

        return super().from_config(config)
