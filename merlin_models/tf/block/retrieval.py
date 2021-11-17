import abc
from collections import deque
from typing import Optional, List, Text, Tuple

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from ..core import (
    Block,
    ParallelBlock,
    TabularAggregation,
    inputs,
    merge,
    tabular_aggregation_registry,
    TabularTransformationsType,
    PredictionTask,
    MetricOrMetricClass,
)
from ..features.embedding import EmbeddingFeatures
from ..typing import TabularData
from ..utils.tf_utils import ContextMixin


class Distance(TabularAggregation, abc.ABC):
    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        assert len(inputs) == 2

        return self.distance(inputs, **kwargs)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()


@tabular_aggregation_registry.register("cosine")
class CosineSimilarity(Distance):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def distance(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        out = self.dot(list(inputs.values()))

        return out


def TwoTowerBlock(
        schema,
        query_tower: Block,
        item_tower: Optional[Block] = None,
        query_tower_tag=Tag.USER,
        item_tower_tag=Tag.ITEM,
        embedding_dim_default: Optional[int] = 64,
        post: Optional[TabularTransformationsType] = None,

        # negative_memory_bank=None,
        **kwargs
) -> ParallelBlock:
    item_tower = item_tower or query_tower.copy()
    two_tower = ParallelBlock(
        {
            str(query_tower_tag): inputs(
                schema.select_by_tag(query_tower_tag),
                query_tower,
                embedding_dim_default=embedding_dim_default,
            ),
            str(item_tower_tag): inputs(
                schema.select_by_tag(item_tower_tag),
                item_tower,
                embedding_dim_default=embedding_dim_default,
            )
        },
        post=post,
        **kwargs
    )

    return two_tower


class PredictionTransformation(tf.keras.layers.Layer, ContextMixin):
    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        return predictions, targets


class SamplingBiasCorrection(PredictionTransformation):
    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        popularity = self.get_from_context("popularity")
        if popularity is not None:
            predictions -= tf.math.log(popularity)

        return predictions, targets


class InBatchNegativeSampling(PredictionTransformation):
    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        scores = tf.linalg.matmul(*list(predictions.values()), transpose_b=True)

        if targets is not None:
            if len(targets.shape) == 2:
                targets = tf.squeeze(targets)
            targets = tf.linalg.diag(targets)
        else:
            targets = tf.eye(tf.shape(scores)[0], tf.shape(scores)[1])

        return scores, targets


class NegativeSampler(abc.ABC):
    @abc.abstractmethod
    def retrieve(self) -> tf.Tensor:
        raise NotImplementedError()


class CrossBatchNegativeSampling(PredictionTransformation):
    def __init__(self, sampler: NegativeSampler):
        self.sampler = sampler

    def call(self, predictions, targets) -> Tuple[tf.Tensor, tf.Tensor]:
        extra_negatives: tf.Tensor = self.sampler.retrieve()
        extra_negatives = array_ops.stop_gradient(extra_negatives,
                                                  name="extra_negatives_stop_gradient")
        predictions = tf.concat([predictions, extra_negatives], axis=0)
        targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        return predictions, targets


class RetrievalPredictionTask(PredictionTask):
    def __init__(self,
                 loss: Optional[tf.keras.losses.Loss] = None,
                 in_batch_negatives: bool = True,
                 extra_negatives: Optional[NegativeSampler] = None,
                 target_name: Optional[str] = None,
                 task_name: Optional[str] = None,
                 metrics: Optional[List[MetricOrMetricClass]] = None, pre: Optional[Layer] = None,
                 prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 name: Optional[Text] = None, **kwargs) -> None:
        loss = loss if loss is not None else tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        super().__init__(loss, target_name, task_name, metrics, pre, None, prediction_metrics,
                         label_metrics, loss_metrics, name, **kwargs)
        self.in_batch_negatives = in_batch_negatives
        self.extra_negatives = extra_negatives

    def compute_loss(
            self,
            predictions,
            targets,
            training: bool = False,
            compute_metrics=True,
            sample_weight: Optional[tf.Tensor] = None,
            **kwargs
    ) -> tf.Tensor:
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]
        if isinstance(predictions, dict) and self.target_name:
            predictions = predictions[self.task_name]

        if self.in_batch_negatives:
            norm_vecs = [tf.linalg.l2_normalize(inp, axis=1) for inp in list(predictions.values())]
            scores = tf.linalg.matmul(*norm_vecs, transpose_b=True)

            if targets is not None:
                if len(targets.shape) == 2:
                    targets = tf.squeeze(targets)
                targets = tf.linalg.diag(targets)
            else:
                targets = tf.eye(tf.shape(scores)[0], tf.shape(scores)[1])
        else:
            if targets is None:
                raise ValueError("Targets are required when in-batch negative sampling is disabled")
            scores = tf.keras.layers.Dot(axes=1, normalize=True)(list(predictions.values()))

        if self.extra_negatives:
            extra_negatives: tf.Tensor = self.extra_negatives.retrieve()
            extra_negatives = array_ops.stop_gradient(extra_negatives,
                                                      name="extra_negatives_stop_gradient")
            scores = tf.concat([scores, extra_negatives], axis=0)
            targets = tf.concat([targets, tf.zeros_like(extra_negatives)], axis=0)

        # Sampling bias correction
        # TODO: add popularity to standard tags
        popularity = self.get_from_context("popularity")
        if popularity is not None:
            scores -= tf.math.log(popularity)

        loss = self.loss(y_true=targets, y_pred=scores, sample_weight=sample_weight)

        if compute_metrics:
            update_ops = self.calculate_metrics(predictions, targets, forward=False, loss=loss)

            update_ops = [x for x in update_ops if x is not None]

            with tf.control_dependencies(update_ops):
                return tf.identity(loss)

        return loss


def MatrixFactorizationBlock(
        schema: Schema,
        dim: int,
        query_id_tag=Tag.USER_ID,
        item_id_tag=Tag.ITEM_ID,
        distance="cosine",
        **kwargs
):
    query_id, item_id = schema.select_by_tag(query_id_tag), schema.select_by_tag(item_id_tag)
    matrix_factorization = merge(
        {
            str(query_id_tag): EmbeddingFeatures.from_schema(query_id, embedding_dim_default=dim),
            str(item_id_tag): EmbeddingFeatures.from_schema(item_id, embedding_dim_default=dim),
        },
        aggregation=distance,
        **kwargs
    )

    return matrix_factorization


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class MemoryBankBlock(Block, NegativeSampler):
    def __init__(
            self,
            num_batches: int = 1,
            key: Optional[str] = None,
            post: Optional[Block] = None,
            no_outputs: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.key = key
        self.num_batches = num_batches
        self.queue = deque(maxlen=num_batches + 1)
        self.no_outputs = no_outputs
        self.post = post

    def call(self, inputs: TabularData, training=True, **kwargs) -> TabularData:
        if training:
            to_add = inputs[self.key] if self.key else inputs
            self.queue.append(to_add)

        if self.no_outputs:
            return {}

        return inputs

    def retrieve(self) -> tf.Tensor:
        batches = list(self.queue)[:-1]
        outputs = tf.concat(batches, axis=0)

        if self.post is not None:
            outputs = self.post(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

# class TwoTower(DualEncoderBlock):
#     def __init__(
#             self,
#             query: Union[tf.keras.layers.Layer, Block],
#             item: Union[tf.keras.layers.Layer, Block],
#             distance: Distance = CosineSimilarity(),
#             pre: Optional[TabularTransformationType] = None,
#             post: Optional[TabularTransformationType] = None,
#             schema: Optional[Schema] = None,
#             name: Optional[str] = None,
#             left_name="query",
#             right_name="item",
#             **kwargs
#     ):
#         super().__init__(
#             query,
#             item,
#             pre=pre,
#             post=post,
#             aggregation=distance,
#             schema=schema,
#             name=name,
#             left_name=left_name,
#             right_name=right_name,
#             **kwargs
#         )
#         self.left_name = left_name
#         self.right_name = right_name
#
#     @classmethod
#     def from_schema(  # type: ignore
#             cls,
#             schema: Schema,
#             dims: List[int],
#             query_tower_tag=Tag.USER,
#             item_tower_tag=Tag.ITEM,
#             **kwargs
#     ) -> "TwoTower":
#         query_tower = inputs(schema.select_by_tag(query_tower_tag), MLPBlock(dims))
#         item_tower = inputs(schema.select_by_tag(item_tower_tag), MLPBlock(dims))
#
#         return cls(query_tower, item_tower, **kwargs)
#
#     @property
#     def query_tower(self) -> Optional[tf.keras.layers.Layer]:
#         if self.left_name in self.parallel_dict:
#             return self.parallel_dict[self.left_name]
#
#         return None
#
#     @property
#     def item_tower(self) -> Optional[tf.keras.layers.Layer]:
#         if self.right_name in self.parallel_dict:
#             return self.parallel_dict[self.right_name]
#
#         return None
#
#
# class MatrixFactorization(DualEncoderBlock):
#     def __init__(
#             self,
#             query_embedding: EmbeddingFeatures,
#             item_embedding: EmbeddingFeatures,
#             distance: Distance = CosineSimilarity(),
#             pre: Optional[TabularTransformationType] = None,
#             post: Optional[TabularTransformationType] = None,
#             schema: Optional[Schema] = None,
#             name: Optional[str] = None,
#             left_name="user",
#             right_name="item",
#             **kwargs
#     ):
#         super().__init__(
#             query_embedding,
#             item_embedding,
#             pre=pre,
#             post=post,
#             aggregation=distance,
#             schema=schema,
#             name=name,
#             left_name=left_name,
#             right_name=right_name,
#             **kwargs
#         )
#
#     @classmethod
#     def from_schema(
#             cls,
#             schema: Schema,
#             query_id_tag=Tag.USER_ID,
#             item_id_tag=Tag.ITEM_ID,
#             distance: Distance = CosineSimilarity(),
#             **kwargs
#     ) -> "MatrixFactorization":
#         query = EmbeddingFeatures.from_schema(schema.select_by_tag(query_id_tag))
#         item = EmbeddingFeatures.from_schema(schema.select_by_tag(item_id_tag))
#
#         return cls(query, item, distance=distance, **kwargs)
