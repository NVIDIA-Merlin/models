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
from typing import Optional, Sequence, Union

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    EmbeddingWithMetadata,
    PredictionOutput,
)
from merlin.models.tf.blocks.core.combinators import ParallelBlock
from merlin.models.tf.blocks.core.context import FeatureContext
from merlin.models.tf.blocks.core.tabular import Filter, TabularAggregationType
from merlin.models.tf.blocks.core.transformations import L2Norm
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.models.base import ModelBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
    rescore_false_negatives,
)
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import Schema

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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DualEncoderBlock(ParallelBlock):
    def __init__(
        self,
        query_block: Block,
        item_block: Block,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        l2_normalization: bool = False,
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
        l2_normalization: bool
            Apply L2 normalization to the user and item representations before
            computing dot interactions.
            Defaults to False.
        """
        if l2_normalization:
            query_block = query_block.connect(L2Norm())
            item_block = item_block.connect(L2Norm())
        self._query_block = TowerBlock(query_block)
        self._item_block = TowerBlock(item_block)

        branches = {
            "query": Filter(query_block.schema).connect(self._query_block),
            "item": Filter(item_block.schema).connect(self._item_block),
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
    store_negative_ids: bool
        Returns negative items ids as part of the output, by default False
    """

    def __init__(
        self,
        samplers: Sequence[ItemSampler] = (),
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value: float = MIN_FLOAT,
        item_id_feature_name: str = "item_id",
        item_domain: str = "item_id",
        query_name: str = "query",
        item_name: str = "item",
        cache_query: bool = False,
        sampled_softmax_mode: bool = False,
        store_negative_ids: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value
        self.item_id_feature_name = item_id_feature_name
        self.item_domain = item_domain
        self.query_name = query_name
        self.item_name = item_name
        self.cache_query = cache_query
        self.store_negative_ids = store_negative_ids

        if not isinstance(samplers, (list, tuple)):
            samplers = (samplers,)  # type: ignore
        self.samplers = samplers
        self.sampled_softmax_mode = sampled_softmax_mode

        self.set_required_features()

    def build(self, input_shapes):
        if isinstance(input_shapes, dict):
            query_shape = input_shapes[self.query_name]
            self.context.add_variable(
                tf.Variable(
                    initial_value=tf.zeros([1, query_shape[-1]], dtype=tf.float32),
                    name="query",
                    trainable=False,
                    validate_shape=False,
                    dtype=tf.float32,
                    shape=tf.TensorShape([None, query_shape[-1]]),
                )
            )

        super().build(input_shapes)

    def _check_input_from_two_tower(self, inputs):
        if set(inputs.keys()) != set([self.query_name, self.item_name]):
            raise ValueError(
                f"Wrong input-names, expected: {[self.query_name, self.item_name]} "
                f"but got: {inputs.keys()}"
            )

    def call(
        self,
        inputs: Union[tf.Tensor, TabularData],
        training: bool = True,
        testing: bool = False,
        **kwargs,
    ) -> Union[tf.Tensor, TabularData]:
        """Based on the user/query embedding (inputs[self.query_name]), uses dot product to score
            the positive item (inputs["item"]).
            For the sampled-softmax mode, logits are computed by multiplying the query vector
            and the item embeddings matrix (self.context.get_embedding(self.item_domain))
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
        if self.cache_query:
            # enabled only during top-k evaluation
            self.context["query"].assign(tf.cast(inputs[self.query_name], tf.float32))

        if training or testing:
            return inputs

        if self.sampled_softmax_mode:
            return self._get_logits_for_sampled_softmax(inputs)

        self._check_input_from_two_tower(inputs)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs[self.query_name], inputs[self.item_name]), keepdims=True, axis=-1
        )
        return positive_scores

    @tf.function
    def call_outputs(
        self,
        outputs: PredictionOutput,
        feature_context: FeatureContext = None,
        training=True,
        testing=False,
        **kwargs,
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
        targets, predictions = outputs.targets, outputs.predictions
        valid_negatives_mask = None

        if self.sampled_softmax_mode or isinstance(targets, tf.Tensor):
            positive_item_ids = targets
        else:
            positive_item_ids = feature_context.features.values[self.item_id_feature_name]

        neg_items_ids = None
        if training or testing:

            assert (
                len(self.samplers) > 0
            ), "At least one sampler is required by ItemRetrievalScorer for negative sampling"

            if self.sampled_softmax_mode:
                predictions = self._prepare_query_item_vectors_for_sampled_softmax(
                    predictions, targets
                )

            batch_items_embeddings = predictions[self.item_name]
            batch_items_metadata = {
                feat_name: feature_context.features.values[feat_name]
                for feat_name in self._required_features
            }

            positive_scores = tf.reduce_sum(
                tf.multiply(predictions[self.query_name], predictions[self.item_name]),
                keepdims=True,
                axis=-1,
            )

            neg_items_embeddings_list = []
            neg_items_ids_list = []

            # Adds items from the current batch into samplers and sample a number of negatives
            for sampler in self.samplers:
                input_data = EmbeddingWithMetadata(batch_items_embeddings, batch_items_metadata)
                sampling_kwargs = {"training": training}
                if "item_weights" in sampler._call_fn_args:
                    sampling_kwargs["item_weights"] = self.context.get_embedding(self.item_domain)
                neg_items = sampler(input_data.__dict__, **sampling_kwargs)

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
                predictions[self.query_name], neg_items_embeddings, transpose_b=True
            )

            if self.downscore_false_negatives or self.store_negative_ids:
                if isinstance(targets, tf.Tensor):
                    positive_item_ids = targets
                else:
                    positive_item_ids = feature_context.features.values[self.item_id_feature_name]

                if len(neg_items_ids_list) == 1:
                    neg_items_ids = neg_items_ids_list[0]
                else:
                    neg_items_ids = tf.concat(neg_items_ids_list, axis=0)

                negative_scores, valid_negatives_mask = rescore_false_negatives(
                    positive_item_ids, neg_items_ids, negative_scores, self.false_negatives_score
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
                negative_item_ids=neg_items_ids,
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
                negative_item_ids=neg_items_ids,
            )

    def _get_logits_for_sampled_softmax(self, inputs):
        if not isinstance(inputs, tf.Tensor):
            raise ValueError(
                f"Inputs to the Sampled Softmax block should be tensors, got {type(inputs)}"
            )
        embedding_table = self.context.get_embedding(self.item_domain)
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
        embedding_table = self.context.get_embedding(self.item_domain)
        batch_items_embeddings = embedding_ops.embedding_lookup(embedding_table, targets)
        predictions = {self.query_name: predictions, self.item_name: batch_items_embeddings}
        return predictions

    def set_required_features(self):
        required_features = set()
        if self.downscore_false_negatives:
            required_features.add(self.item_id_feature_name)

        required_features.update(
            [feature for sampler in self.samplers for feature in sampler.required_features]
        )

        self._required_features = list(required_features)

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
