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
from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import embedding_ops

from merlin.schema import Schema, Tags

from ...utils.constants import MIN_FLOAT
from ...utils.schema import categorical_cardinalities
from ..blocks.transformations import L2Norm
from ..core import Block, EmbeddingWithMetadata
from ..losses.loss_base import LossType
from ..metrics.ranking import ranking_metrics
from ..prediction.sampling import InBatchSampler, ItemSampler, PopularityBasedSampler
from ..typing import TabularData
from ..utils.tf_utils import maybe_deserialize_keras_objects, maybe_serialize_keras_objects
from .classification import CategFeaturePrediction, MultiClassClassificationTask

LOG = logging.getLogger("merlin.models")


@Block.registry.register_with_multiple_names("sampling-bias-correction")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictionsScaler(Block):
    def __init__(self, scale_factor: float, **kwargs):
        super(PredictionsScaler, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        if not training:
            return inputs * self.scale_factor
        else:
            return inputs

    def call_targets(self, predictions, targets, training=True, **kwargs) -> tf.Tensor:
        if training:
            return targets, predictions * self.scale_factor
        return targets

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config["scale_factor"] = self.scale_factor

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ItemsPrediction(CategFeaturePrediction):
    def __init__(
        self,
        schema: Schema,
        **kwargs,
    ):
        super(ItemsPrediction, self).__init__(schema, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ItemsPredictionWeightTying(Block):
    def __init__(self, schema: Schema, bias_initializer="zeros", **kwargs):
        super(ItemsPredictionWeightTying, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        self.item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.num_classes = categorical_cardinalities(schema)[self.item_id_feature_name]

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )
        return super().build(input_shape)

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        embedding_table = self.context.get_embedding(self.item_id_feature_name)
        logits = tf.matmul(inputs, embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        logits = tf.cast(logits, tf.float32)

        return logits


# TODO: Implement this for the MIND prediction: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(Block):
    def predict(
        self, predictions, targets=None, training=True, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")


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
    """

    def __init__(
        self,
        samplers: Sequence[ItemSampler] = (),
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value: int = MIN_FLOAT,
        item_id_feature_name: str = "item_id",
        query_name: str = "query",
        item_name: str = "item",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.downscore_false_negatives = sampling_downscore_false_negatives
        self.false_negatives_score = sampling_downscore_false_negatives_value
        self.item_id_feature_name = item_id_feature_name
        self.samplers = samplers
        self.query_name = query_name
        self.item_name = item_name

        if not isinstance(self.samplers, (list, tuple)):
            self.samplers = (self.samplers,)

        self.set_required_features()

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
        if set(inputs.keys()) != set([self.query_name, self.item_name]):
            raise ValueError(
                f"Wrong input-names, expected: {[self.query_name, self.item_name]} "
                f"but got: {inputs.keys()}"
            )

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
        self, inputs: Union[tf.Tensor, TabularData], training: bool = True, **kwargs
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
        if training:
            return inputs

        if isinstance(inputs, tf.Tensor):
            embedding_table = self.context.get_embedding(self.item_id_feature_name)
            all_scores = tf.matmul(inputs, tf.transpose(embedding_table))
            return all_scores

        self._check_input_from_two_tower(inputs)
        positive_scores = tf.reduce_sum(
            tf.multiply(inputs[self.query_name], inputs[self.item_name]), keepdims=True, axis=-1
        )
        return positive_scores

    @tf.function
    def call_targets(self, predictions, targets, training=True, **kwargs) -> tf.Tensor:
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
        self._check_required_context_item_features_are_present()

        if training:
            assert (
                len(self.samplers) > 0
            ), "At least one sampler is required by ItemRetrievalScorer for negative sampling"

            if isinstance(predictions, dict):
                batch_items_embeddings = predictions[self.item_name]
            else:
                embedding_table = self.context.get_embedding(self.item_id_feature_name)
                batch_items_embeddings = embedding_ops.embedding_lookup(embedding_table, targets)
                predictions = {self.query_name: predictions, self.item_name: batch_items_embeddings}
            batch_items_metadata = self.get_batch_items_metadata()

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
                predictions[self.query_name], neg_items_embeddings, transpose_b=True
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

            # To ensure that the output is always fp32, avoiding numerical
            # instabilities with mixed_float16 policy
            predictions = tf.cast(predictions, tf.float32)

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
        return targets, predictions

    def get_batch_items_metadata(self):
        result = {feat_name: self.context[feat_name] for feat_name in self._required_features}
        return result

    def rescore_false_negatives(self, positive_item_ids, neg_samples_item_ids, negative_scores):
        # Removing dimensions of size 1 from the shape of the item ids, if applicable
        positive_item_ids = tf.squeeze(positive_item_ids)
        neg_samples_item_ids = tf.squeeze(neg_samples_item_ids)

        # Reshapes positive and negative ids so that false_negatives_mask matches the scores shape
        false_negatives_mask = tf.equal(
            tf.expand_dims(positive_item_ids, -1), tf.expand_dims(neg_samples_item_ids, 0)
        )

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


def ItemRetrievalTask(
    schema: Schema,
    loss: Optional[LossType] = "categorical_crossentropy",
    samplers: Sequence[ItemSampler] = (),
    metrics=ranking_metrics(top_ks=[10, 20]),
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    softmax_temperature: float = 1,
    normalize: bool = True,
    query_name: str = "query",
    item_name: str = "item",
) -> MultiClassClassificationTask:
    """
    Function to create the ItemRetrieval task with the right parameters.

    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        loss: Optional[LossType]
            Loss function.
            Defaults to `categorical_crossentropy`.
        samplers: List[ItemSampler]
            List of samplers for negative sampling, by default `[InBatchSampler()]`
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
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    if samplers is None or len(samplers) == 0:
        samplers = (InBatchSampler(),)

    prediction_call = ItemRetrievalScorer(
        samplers=samplers,
        item_id_feature_name=item_id_feature_name,
        query_name=query_name,
        item_name=item_name,
    )

    if normalize:
        prediction_call = L2Norm().connect(prediction_call)

    if softmax_temperature != 1:
        prediction_call = prediction_call.connect(PredictionsScaler(1.0 / softmax_temperature))

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


@Block.registry.register_with_multiple_names("remove_pad_3d")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0

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


@Block.registry.register_with_multiple_names("masking_head")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MaskingHead(Block):
    """
    The masking class to transform targets based on the
    boolean masking schema stored in the model's context
    Parameters
    ----------
        padding_idx: int
            The padding index value.
            Defaults to 0.
        item_id_feature_name: str
            Name of the column containing the item ids
            Defaults to `item_id`
    Returns
    -------
        targets: tf.Tensor
            Tensor of masked labels.
    """

    def __init__(self, item_id_feature_name: str = "item_id", **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0
        self.item_id_feature_name = item_id_feature_name

    def call_targets(
        self, predictions: tf.Tensor, targets: tf.Tensor, training: bool = True, **kwargs
    ) -> tf.Tensor:
        targets = self.context[self.item_id_feature_name]
        mask = self.context.get_mask()
        targets = tf.where(mask, targets, self.padding_idx)
        return targets


def ItemsPredictionSampled(
    schema: Schema,
    num_sampled: int,
    min_id: int = 0,
    ignore_false_negatives: bool = True,
):
    """
    Compute the items logits on a subset of sampled candidates to optimize
    training. During inference, the scores are computed over the whole
    catalog of items.
    Reference of the method can be found at [Jean et al., 2014](http://arxiv.org/abs/1412.2007)

    Parameters:
    -----------
        schema: Schema
            The schema object including features to use and their properties.
        num_sampled: int
            The number of candidates to sample during training
        min_id: int
            The minimum id value to be sampled as negative. Useful to ignore the first categorical
            encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
            Defaults to 0.
        ignore_false_negatives: bool
            Ignore sampled items that are equal to the target classes
            Defaults to True

    Returns:
    -------
        targets, logits: tf.Tensor, tf.Tensor
            During training, return the concatenated tensor of true class
            and sampled negatives of shape (bs, num_sampled+1), as well as the related logits.
            During evaluation, returns the input tensor of true class, and the related logits.
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    num_classes = categorical_cardinalities(schema)[item_id_feature_name]
    samplers = PopularityBasedSampler(
        max_num_samples=num_sampled,
        max_id=num_classes,
        min_id=min_id,
        item_id_feature_name=item_id_feature_name,
    )

    logits = ItemRetrievalScorer(
        samplers=samplers,
        sampling_downscore_false_negatives=ignore_false_negatives,
        item_id_feature_name=item_id_feature_name,
    )

    return logits


def NextItemPredictionTask(
    schema: Schema,
    loss: Optional[LossType] = "sparse_categorical_crossentropy",
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
    min_sampled_id: int = 0,
) -> MultiClassClassificationTask:
    """
    Function to create the NextItemPrediction task with the right parameters.
    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        loss: Optional[LossType]
            Loss function.
            Defaults to `sparse_categorical_crossentropy`.
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
            When set to True, loss should be set to
            `tf.keras.losses.CategoricalCrossentropy(from_logits=True)`
            and metrics to `ranking_metrics(top_ks=..., labels_onehot=False)`
            Defaults to False
        num_sampled: int
            When sampled_softmax is enabled, specify the number of
            negative candidates to generate for each batch
            Defaults to 100
        min_sampled_id: int
            The minimum id value to be sampled. Useful to ignore the first categorical
            encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
            Defaults to 0.
    Returns
    -------
        PredictionTask
            The next item prediction task
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    if sampled_softmax:
        prediction_call = ItemsPredictionSampled(
            schema, num_sampled=num_sampled, min_id=min_sampled_id
        )

    elif weight_tying:
        prediction_call = ItemsPredictionWeightTying(schema)

    else:
        prediction_call = ItemsPrediction(schema)

    if softmax_temperature != 1:
        prediction_call = prediction_call.connect(PredictionsScaler(1.0 / softmax_temperature))

    if masking:
        prediction_call = MaskingHead(item_id_feature_name=item_id_feature_name).connect(
            RemovePad3D(), prediction_call
        )

    if normalize:
        prediction_call = L2Norm().connect(prediction_call)

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
