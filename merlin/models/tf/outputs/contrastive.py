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
import warnings
from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

import tensorflow as tf
from tensorflow.keras.layers import Layer

import merlin.io
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.outputs.base import DotProduct, MetricsFn, ModelOutput
from merlin.models.tf.outputs.classification import (
    CategoricalTarget,
    EmbeddingTablePrediction,
    default_categorical_prediction_metrics,
)
from merlin.models.tf.outputs.sampling.base import (
    Candidate,
    ItemSamplersType,
    parse_negative_samplers,
)
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.constants import MIN_FLOAT
from merlin.schema import ColumnSchema, Schema

LOG = logging.getLogger("merlin_models")


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContrastiveOutput(ModelOutput):
    """Categorical output

    Parameters
    ----------
    to_call: Union[Schema, ColumnSchema,
                EmbeddingTable, 'CategoricalTarget',
                'EmbeddingTablePrediction', 'DotProduct']
        The target feature to predict. To perform weight-tying [1] technique, you should provide
        the `EmbeddingTable` or `EmbeddingTablePrediction` related to the
        target feature.
    negative_samplers: ItemSamplersType
        List of samplers for negative sampling,
        by default None
    pre: Optional[Block], optional
        Optional block to transform predictions before computing the binary logits,
        by default None
    post: Optional[Block], optional
        Optional block to transform the binary logits,
        by default None
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1
    name: str, optional
        The name of the task, by default None
    default_loss: Union[str, tf.keras.losses.Loss], optional
        Default loss to use for categorical-classification
        by default 'categorical_crossentropy'
    get_default_metrics: Callable, optional
        A function returning the list of default metrics
        to use for categorical-classification
    store_negative_ids: bool, optional
        Whether to store negative ids for post-processing
        by default False
    logq_sampling_correction: bool, optional
        The LogQ correction is a standard technique for
        sampled softmax and popularity-biased sampling.
        It subtracts from the logits the
        log expected count/prob of the positive and
        negative samples in order to not overpenalize the
        popular items for being sampled more often as negatives.
        It can be enabled if a single negative sampler is provided
        and if it provides the sampler provides the
        sampling probabilities (i.e. implements with_sampling_probs()).
        Another alternative for performing logQ correction is using
        ContrastiveOutput(..., post=PopularityLogitsCorrection(item_frequencies)),
        where you need to provide the items frequency probability distribution (prior).
        Default is False.

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).

    Notes:
    ----------
    In case `to_call` is set as `DotProduct()`, schema of target couldn't be inferred
    therefore, the user should feed a schema only with ITEM_ID feature as schema arg,
    which is treated as a `kwargs` arg below.

    Example usage::
        outputs=mm.ContrastiveOutput(
            to_call=DotProduct(),
            negative_samplers="in-batch",
            schema=schema.select_by_tag(Tags.ITEM_ID),
            logits_temperature = 0.2,
        )

    The schema arg is not needed when we pass the schema to `to_call` arg.

    Example usage::
        outputs=mm.ContrastiveOutput(
            to_call=schema.select_by_tag(Tags.ITEM_ID),
            negative_samplers="in-batch",
            logits_temperature = 0.2,
        )

    """

    def __init__(
        self,
        to_call: Union[
            Schema,
            ColumnSchema,
            EmbeddingTable,
            CategoricalTarget,
            EmbeddingTablePrediction,
            DotProduct,
        ],
        negative_samplers: ItemSamplersType,
        target_name: str = None,
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
        default_loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
        default_metrics_fn: MetricsFn = default_categorical_prediction_metrics,
        downscore_false_negatives=True,
        false_negative_score: float = MIN_FLOAT,
        query_name: str = "query",
        candidate_name: str = "candidate",
        store_negative_ids: bool = False,
        logq_sampling_correction: Optional[bool] = False,
        **kwargs,
    ):
        self.col_schema = None
        _to_call = None
        if to_call is not None:
            if isinstance(to_call, (Schema, ColumnSchema)):
                if isinstance(to_call, Schema):
                    if len(to_call) == 1:
                        to_call = to_call.first
                    else:
                        raise ValueError("to_call must be a single column schema")

                self.col_schema = to_call
                _to_call = CategoricalTarget(to_call)
                target_name = target_name or to_call.name
            elif isinstance(to_call, EmbeddingTable):
                _to_call = EmbeddingTablePrediction(to_call)
                target_name = _to_call.table.col_schema.name
                self.col_schema = _to_call.table.col_schema
            else:
                _to_call = to_call

        if "schema" in kwargs:
            self.col_schema = kwargs.pop("schema").first

        if not self.col_schema:
            raise ValueError(
                "schema of target couldn't be inferred, please provide ", "`schema=...`"
            )

        self.negative_samplers = parse_negative_samplers(negative_samplers)
        self.downscore_false_negatives = downscore_false_negatives
        self.false_negative_score = false_negative_score
        self.query_name = query_name
        self.candidate_name = candidate_name
        self.store_negative_ids = store_negative_ids
        self.logq_sampling_correction = logq_sampling_correction

        self.target_name = kwargs.pop("target", target_name)
        super().__init__(
            to_call=_to_call,
            default_loss=default_loss,
            default_metrics_fn=default_metrics_fn,
            name=name,
            target=self.target_name,
            pre=pre,
            post=post,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def build(self, input_shape):
        if (
            isinstance(input_shape, dict)
            and all(key in input_shape for key in self.keys)
            and not isinstance(self.to_call, DotProduct)
        ):
            self.to_call = DotProduct(*self.keys)

        super().build(input_shape)

    def call(self, inputs, features=None, targets=None, training=False, testing=False):
        call_kwargs = dict(features=features, targets=targets, training=training, testing=testing)

        if training or testing:
            if self.has_candidate_weights and targets is None:
                return tf_utils.call_layer(self.to_call, inputs, **call_kwargs)

            return self.call_contrastive(inputs, **call_kwargs)

        return tf_utils.call_layer(self.to_call, inputs, **call_kwargs)

    def call_contrastive(self, inputs, features, targets, training=False, testing=False):
        if isinstance(inputs, dict) and self.query_name in inputs:
            query_embedding = inputs[self.query_name]
        elif isinstance(inputs, tf.Tensor):
            query_embedding = inputs
        else:
            raise ValueError("Couldn't infer query embedding")

        if self.has_candidate_weights:
            positive_id = targets
            if isinstance(targets, dict):
                positive_id = targets[self.col_schema.name]
            positive_embedding = self.embedding_lookup(positive_id)
        else:
            positive_id = features[self.col_schema.name]
            positive_embedding = inputs[self.candidate_name]

        positive = Candidate(id=positive_id, metadata={**features}).with_embedding(
            positive_embedding
        )
        negative, positive = self.sample_negatives(
            positive, features, training=training, testing=testing
        )
        if self.has_candidate_weights and (
            positive.id.shape != negative.id.shape or positive != negative
        ):
            negative = negative.with_embedding(self.embedding_lookup(negative.id))

        return self.outputs(query_embedding, positive, negative)

    def outputs(
        self, query_embedding: tf.Tensor, positive: Candidate, negative: Candidate
    ) -> Prediction:
        """Method to compute the dot product between the query embeddings and
        positive/negative candidates

        Parameters
        ----------
        query_embedding : tf.Tensor
            tensor of query embeddings.
        positive : Candidate
            Store the ids and metadata (such as embeddings) of the positive candidates.
        negative : Candidate
            Store the ids and metadata (such as embeddings) of the sampled negative candidates.

        Returns
        -------
        Prediction
            a Prediction object with the prediction scores, the targets and
            the negative candidates ids if specified.
        """
        if not positive.has_embedding:
            raise ValueError("Positive candidate must have an embedding")
        if not negative.has_embedding:
            raise ValueError("Negative candidate must have an embedding")

        # Apply dot-product
        negative_scores = tf.linalg.matmul(query_embedding, negative.embedding, transpose_b=True)

        positive_scores = tf.reduce_sum(
            tf.multiply(query_embedding, positive.embedding), keepdims=True, axis=-1
        )

        if self.logq_sampling_correction:
            if positive.sampling_prob is None or negative.sampling_prob is None:
                warnings.warn(
                    "The logQ sampling correction is enabled, but sampling probs were not found "
                    "for both positive and negative candidates",
                    RuntimeWarning,
                )

            epsilon = 1e-16
            positive_scores -= tf.math.log(positive.sampling_prob + epsilon)
            negative_scores -= tf.math.log(tf.transpose(negative.sampling_prob + epsilon))

        if self.downscore_false_negatives:
            negative_scores, _ = tf_utils.rescore_false_negatives(
                positive.id, negative.id, negative_scores, self.false_negative_score
            )

        outputs = tf.concat([positive_scores, negative_scores], axis=-1)

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        outputs = tf.cast(outputs, tf.float32)

        targets = tf.concat(
            [
                tf.ones([tf.shape(outputs)[0], 1], dtype=outputs.dtype),
                tf.zeros(
                    [tf.shape(outputs)[0], tf.shape(outputs)[1] - 1],
                    dtype=outputs.dtype,
                ),
            ],
            axis=1,
        )
        if self.store_negative_ids:
            return Prediction(outputs, targets, negative_candidate_ids=negative.id)
        return Prediction(outputs, targets)

    def sample_negatives(
        self,
        positive: Candidate,
        features: TabularData,
        training=False,
        testing=False,
    ) -> Tuple[Candidate, Candidate]:
        """Method to sample negatives from `self.negative_samplers`

        Parameters
        ----------
        positive_items : Items
            Positive items (ids and metadata)
        features : TabularData
            Dictionary of input raw tensors
        training : bool, optional
            Flag for train mode, by default False
        testing : bool, optional
            Flag for test mode, by default False

        Returns
        -------
        Tuple[Candidate, Candidate]
            Tuple of candidates with sampled negative ids and the provided positive ids
            added with the sampling probability
        """
        sampling_kwargs = {"training": training, "testing": testing, "features": features}
        candidates: List[Candidate] = []

        if self.logq_sampling_correction and len(self.negative_samplers) > 1:
            raise ValueError(
                "It is only possible to apply logQ sampling correction "
                "(logq_sampling_correction=True) when only one negative sampler is provided."
            )

        for sampler in self.negative_samplers:
            neg_samples: Candidate = tf_utils.call_layer(sampler, positive, **sampling_kwargs)

            # Adds to the positive and negative candidates their sampling probs from the sampler
            positive = sampler.with_sampling_probs(positive)
            neg_samples = sampler.with_sampling_probs(neg_samples)

            if neg_samples.id is not None:
                candidates.append(neg_samples)
            else:
                LOG.warn(
                    f"The sampler {type(sampler).__name__} returned no samples for this batch."
                )

        if len(candidates) == 0:
            raise Exception(
                f"No negative items where sampled from samplers {self.negative_samplers}"
            )

        negatives = candidates[0]
        if len(candidates) > 1:
            for neg in candidates[1:]:
                negatives += neg

        return negatives, positive

    def embedding_lookup(self, ids: tf.Tensor):
        return self.to_call.embedding_lookup(tf.squeeze(ids))

    def to_dataset(self, gpu=None) -> merlin.io.Dataset:
        return merlin.io.Dataset(tf_utils.tensor_to_df(self.to_call.embeddings, gpu=gpu))

    @property
    def has_candidate_weights(self) -> bool:
        if isinstance(self.to_call, DotProduct):
            return False

        return isinstance(self.to_call, LookUpProtocol)

    @property
    def keys(self) -> List[str]:
        return [self.query_name, self.candidate_name]

    def set_negative_samplers(self, negative_samplers: ItemSamplersType):
        if negative_samplers is not None:
            negative_samplers = parse_negative_samplers(negative_samplers)
        self.negative_samplers = negative_samplers

    def get_config(self):
        config = super().get_config()

        config = tf_utils.maybe_serialize_keras_objects(self, config, ["negative_samplers"])

        config["target"] = self.target_name
        config["downscore_false_negatives"] = self.downscore_false_negatives
        config["false_negative_score"] = self.false_negative_score
        config["query_name"] = self.query_name
        config["candidate_name"] = self.candidate_name
        config["store_negative_ids"] = self.store_negative_ids

        config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(
            Schema([self.col_schema])
        )

        return config

    @classmethod
    def from_config(cls, config):
        config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        config = tf_utils.maybe_deserialize_keras_objects(config, ["negative_samplers"])

        return super().from_config(config)


@runtime_checkable
class LookUpProtocol(Protocol):
    """Protocol for embedding lookup layers"""

    @property
    def embeddings(self):
        pass

    def embedding_lookup(self, inputs, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass
