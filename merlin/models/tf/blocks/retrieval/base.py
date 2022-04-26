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
from typing import List, Optional, Sequence, Union, TypedDict, Callable

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    EmbeddingWithMetadata,
    PredictionOutput,
)
from merlin.models.tf.blocks.core.combinators import ParallelBlock
from merlin.models.tf.blocks.core.tabular import Filter, TabularAggregationType
from merlin.models.tf.blocks.core.transformations import RenameFeatures
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.models.base import ModelBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import Schema, Tags

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TowerBlock(ModelBlock):
    """TowerBlock to wrap item or query tower"""

    pass


class RetrievalMixin:
    def query_block(self) -> TowerBlock:
        """Method to return the query tower from a RetrievalModel instance"""
        raise NotImplementedError()

    def item_block(self) -> TowerBlock:
        """Method to return the item tower from a RetrievalModel instance"""
        raise NotImplementedError()


TensorOrCallable = Union[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]


class DualEncoderOutputs(TypedDict):
    """Outputs from a DualEncoderBlock"""

    query: tf.Tensor
    query_id: tf.Tensor
    item: TensorOrCallable
    item_id: tf.Tensor
    negative_item: Optional[TensorOrCallable]
    negative_item_id: Optional[tf.Tensor]


class ContrastiveInputs(TypedDict):
    """Outputs from a DualEncoderBlock"""

    query: tf.Tensor
    query_id: tf.Tensor
    item: TensorOrCallable
    item_id: tf.Tensor
    negative_item: Optional[TensorOrCallable]
    negative_item_id: Optional[tf.Tensor]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DualEncoderBlock(ParallelBlock):
    QUERY_BRANCH_NAME = "query"
    ITEM_BRANCH_NAME = "item"

    def __init__(
            self,
            query_block: Block,
            item_block: Block,
            query_id_tag=Tags.USER_ID,
            item_id_tag=Tags.ITEM_ID,
            output_ids: bool = True,
            pre: Optional[BlockType] = None,
            post: Optional[BlockType] = None,
            aggregation: Optional[TabularAggregationType] = None,
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            strict: bool = False,
            **kwargs,
    ):
        """Prepare the Query and Item towers of a Retrieval block

        Parameters
        ----------
        query_block : Block
            The `Block` instance that combines user features
        item_block : Block
            Optional `Block` instance that combines items features.
        pre : Optional[BlockType], optional
            Optional `Block` instance to apply before the `call` method of the Two-Tower block
        post : Optional[BlockType], optional
            Optional `Block` instance to apply on both outputs of Two-tower model
        aggregation : Optional[TabularAggregationType], optional
            The Aggregation operation to apply after processing the `call` method
            to output a single Tensor.
        schema : Optional[Schema], optional
            The `Schema` object with the input features.
        name : Optional[str], optional
            Name of the layer.
        strict : bool, optional
            If enabled, check that the input of the ParallelBlock instance is a dictionary.
        """
        self._query_block = TowerBlock(query_block)
        self._item_block = TowerBlock(item_block)

        if output_ids:
            query_id = query_block.schema.select_by_tag(query_id_tag)
            item_id = item_block.schema.select_by_tag(item_id_tag)

            if not query_id:
                raise ValueError(f"No feature with tag {query_id_tag} in schema")
            if not item_id:
                raise ValueError(f"No feature with tag {item_id_tag} in schema")

            query_filter = Filter(query_id).connect(
                RenameFeatures({query_id.first.name: "query_id"})
            )
            query_branch = Filter(query_block.schema).connect_with_shortcut(
                self._query_block,
                shortcut_filter=query_filter,
                block_outputs_name="query",
                shortcut_name="query_id",
            )

            item_filter = Filter(item_id).connect(RenameFeatures({item_id.first.name: "item_id"}))
            item_branch = Filter(item_block.schema).connect_with_shortcut(
                self._item_block,
                shortcut_filter=item_filter,
                block_outputs_name="item",
                shortcut_name="item_id",
            )
        else:
            query_branch = Filter(query_block.schema).connect(self._query_block)
            item_branch = Filter(item_block.schema).connect(self._item_block)

        branches = {
            self.QUERY_BRANCH_NAME: query_branch,
            self.ITEM_BRANCH_NAME: item_branch,
        }

        super().__init__(
            branches,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    def query_block(self) -> TowerBlock:
        return self._query_block

    def item_block(self) -> TowerBlock:
        return self._item_block

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)
        output = ParallelBlock(inputs, **config)
        output.__class__ = cls

        return output


@Block.registry.register_with_multiple_names("item_retrieval_scorer")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
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
    query_name: str
        Identify query tower for query/user embeddings, by default 'query'
    item_name: str
        Identify item tower for item embeddings, by default'item'
    cache_query: bool
        Add query embeddings to the context block, by default False
    sampled_softmax_mode: bool
        Use sampled softmax for scoring, by default False
    """

    def __init__(
            self,
            sampling_downscore_false_negatives=True,
            sampling_downscore_false_negatives_value: float = MIN_FLOAT,
            item_id_feature_name: str = "item_id",
            cache_query: bool = False,
            sampled_softmax_mode: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value
        self.item_id_feature_name = item_id_feature_name
        self.cache_query = cache_query
        self.sampled_softmax_mode = sampled_softmax_mode

    # def build(self, input_shapes):
    #     if isinstance(input_shapes, dict):
    #         query_shape = input_shapes["query"]
    #         self.context.add_variable(
    #             tf.Variable(
    #                 initial_value=tf.zeros([1, query_shape[-1]], dtype=tf.float32),
    #                 name="query",
    #                 trainable=False,
    #                 validate_shape=False,
    #                 dtype=tf.float32,
    #                 shape=tf.TensorShape([None, query_shape[-1]]),
    #             )
    #         )
    #         self.context.add_variable(
    #             tf.Variable(
    #                 initial_value=tf.zeros([1, query_shape[-1]], dtype=tf.float32),
    #                 name="positive_candidates_embeddings",
    #                 trainable=False,
    #                 validate_shape=False,
    #                 dtype=tf.float32,
    #                 shape=tf.TensorShape([None, query_shape[-1]]),
    #             )
    #         )
    #     super().build(input_shapes)

    def _check_input_from_two_tower(self, inputs):
        if not all(to_check in set(inputs.keys()) for to_check in ["query", "item"]):
            raise ValueError(
                f"Wrong input-names, expected: query, item "
                f"but got: {inputs.keys()}"
            )

    # def _check_required_context_item_features_are_present(self):
    #     not_found = list(
    #         [
    #             feat_name
    #             for feat_name in self._required_features
    #             if getattr(self, "_context", None) is None
    #                or feat_name not in self.context.named_variables
    #         ]
    #     )
    #
    #     if len(not_found) > 0:
    #         raise ValueError(
    #             f"The following required context features should be available "
    #             f"for the samplers, but were not found: {not_found}"
    #         )

    def call(
            self,
            inputs: DualEncoderOutputs,
            training: bool = False,
            testing: bool = False,
            **kwargs,
    ) -> Union[tf.Tensor, TabularData]:
        """Based on the user/query embedding (inputs[self.query_name]), uses dot product to score
            the positive item (inputs["item"]).
            For the sampled-softmax mode, logits are computed by multiplying the query vector
            and the item embeddings matrix (self.context.get_embedding(self.item_column)))
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
        # if self.cache_query:
        #     # enabled only during top-k evaluation
        #     self.context["query"].assign(tf.cast(inputs[self.query_name], tf.float32))
        #     self.context["positive_candidates_embeddings"].assign(
        #         tf.cast(inputs[self.item_name], tf.float32)
        #     )

        if training or testing:
            return inputs

        if self.sampled_softmax_mode:
            return self._get_logits_for_sampled_softmax(inputs)

        self._check_input_from_two_tower(inputs)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
        )
        return positive_scores

    # @tf.function
    def call_outputs(
            self, outputs: PredictionOutput, training=True, testing=False, **kwargs
    ) -> "PredictionOutput":
        """Based on the user/query embedding (inputs[self.query_name]), uses dot product to score
            the positive item and also sampled negative items (during training).
        Parameters
        ----------
        inputs : TabularData
            Dict with the query and item embeddings (e.g. `{"query": <emb>}, "item": <emb>}`),
            where embeddings are 2D tensors (batch size, embedding size)
        training : bool, optional
            Flag that indicates whether in training mode, by default True
        Returns
        -------
        [tf.Tensor,tf.Tensor]
            all_scores: 2D Tensor with the scores for the positive items and, if `training=True`,
            for the negative sampled items too.
            Return tensor is 2D (batch size, 1 + #negatives)
        """
        valid_negatives_mask = None
        predictions: DualEncoderOutputs = outputs.predictions
        targets = outputs.targets

        if self.sampled_softmax_mode or isinstance(targets, tf.Tensor):
            positive_item_ids = targets
        else:
            positive_item_ids = predictions["item_id"]

        positive_scores = tf.reduce_sum(
            tf.multiply(predictions["query"], predictions["item"]),
            keepdims=True,
            axis=-1,
        )

        if "negative_item" not in predictions:
            return PredictionOutput(positive_scores, targets)

        if training or testing:
            if self.sampled_softmax_mode:
                predictions = self._prepare_query_item_vectors_for_sampled_softmax(
                    predictions, targets
                )

            positive_scores = tf.reduce_sum(
                tf.multiply(predictions["query"], predictions["item"]),
                keepdims=True,
                axis=-1,
            )

            negative_scores = tf.linalg.matmul(
                predictions["query"], predictions["negative_item"], transpose_b=True
            )

            if self.downscore_false_negatives and "negative_item_id" in predictions:
                negative_scores, valid_negatives_mask = rescore_false_negatives(
                    positive_item_ids,
                    predictions["negative_item_id"],
                    negative_scores,
                    self.false_negatives_score
                )

            predictions = tf.concat([positive_scores, negative_scores], axis=-1)

            # To ensure that the output is always fp32, avoiding numerical
            # instabilities with mixed_float16 policy
            predictions = tf.cast(predictions, tf.float32)

        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
        # prepare targets for computing the loss and metrics
        if self.sampled_softmax_mode and not training:
            # Converts target ids to one-hot representation
            num_classes = tf.shape(predictions)[-1]
            targets_one_hot = tf.one_hot(tf.reshape(targets, (-1,)), num_classes)
            return PredictionOutput(
                predictions,
                targets_one_hot,
                positive_item_ids=positive_item_ids,
                valid_negatives_mask=valid_negatives_mask,
            )
        else:
            # Positives in the first column and negatives in the subsequent columns
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
            return PredictionOutput(
                predictions,
                targets,
                positive_item_ids=positive_item_ids,
                valid_negatives_mask=valid_negatives_mask,
            )

    def _get_logits_for_sampled_softmax(self, inputs):
        if not isinstance(inputs, tf.Tensor):
            raise ValueError(
                f"Inputs to the Sampled Softmax block should be tensors, got {type(inputs)}"
            )
        embedding_table = self.context.get_embedding(self.item_id_feature_name)
        all_scores = tf.matmul(inputs, tf.transpose(embedding_table))
        return all_scores

    def _prepare_query_item_vectors_for_sampled_softmax(
            self, predictions: tf.Tensor, targets: tf.Tensor
    ):
        # extract positive items embeddings
        if not isinstance(predictions, tf.Tensor):
            raise ValueError(
                f"Inputs to the Sampled Softmax block should be tensors, got {type(predictions)}"
            )
        embedding_table = self.context.get_embedding(self.item_id_feature_name)
        batch_items_embeddings = embedding_ops.embedding_lookup(embedding_table, targets)
        predictions = {"query": predictions, "item": batch_items_embeddings}
        return predictions

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
