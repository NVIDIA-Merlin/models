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
from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import tensorflow as tf
from merlin.models.tf.blocks.sampling.cross_batch import PopularityBasedSampler

from merlin.models.tf.blocks.sampling.base import ItemSampler
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.layers import Dense
from tensorflow.python.ops import embedding_ops, math_ops, sparse_ops, gen_math_ops, standard_ops, nn_ops

from merlin.models.tf.dataset import DictWithSchema
from merlin.models.tf.losses import LossType, loss_registry
from merlin.models.tf.blocks.core.base import Block, MetricOrMetrics, PredictionOutput, TaskResults
from merlin.models.tf.blocks.core.transformations import LogitsTemperatureScaler, RemovePad3D, L2Norm, remove_pad_3d, \
    remove_pad_3d_targets
from merlin.models.tf.prediction_tasks.base import PredictionTask
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects, transform_label_to_onehot,
)
from merlin.models.utils.schema_utils import categorical_cardinalities
from merlin.schema import Schema, Tags

if TYPE_CHECKING:
    from merlin.models.tf.prediction_tasks.contrastive import ContrastiveLearningTask


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BinaryClassificationTask(PredictionTask):
    """
    Prediction task for binary classification.

    Parameters
    ----------
    target: Union[str, Schema], optional
        The name of the target. If a Schema is provided, the target is inferred from the schema.
    task_name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    loss: LossType, optional
        The loss to use for the task.
        Defaults to "binary_crossentropy".
    metrics: MetricOrMetrics, optional
        The metrics to use for the task. Defaults to [precision, recall, accuracy & auc].
    """

    # Default loss to use
    DEFAULT_LOSS = "binary_crossentropy"

    # Default metrics to use
    DEFAULT_METRICS = (
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.AUC,
    )

    def __init__(
        self,
        target: Optional[Union[str, Schema]] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics: Optional[MetricOrMetrics] = DEFAULT_METRICS,
        **kwargs,
    ):
        if isinstance(target, Schema):
            target_name = target.select_by_tag(Tags.BINARY_CLASSIFICATION)
            if not target_name.column_names:
                raise ValueError(
                    "Binary classification task requires a column with a ",
                    "`Tags.BINARY_CLASSIFICATION` tag.",
                )
            elif len(target_name.column_names) > 1:
                raise ValueError(
                    "Binary classification task requires a single column with a ",
                    "`Tags.BINARY_CLASSIFICATION` tag. ",
                    "Found {} columns. ".format(len(target_name.column_names)),
                    "Please specify the column name with the `target` argument.",
                )
            target_name = target_name.column_names[0]
        else:
            target_name = target if target else kwargs.pop("target_name", None)

        output_layer = kwargs.pop("output_layer", None)
        super().__init__(
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            **kwargs,
        )

        self.output_layer = output_layer or tf.keras.layers.Dense(
            1, activation="linear", name=self.child_name("output_layer")
        )
        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 (fp16) policy
        self.output_activation = tf.keras.layers.Activation(
            "sigmoid", dtype="float32", name="prediction"
        )
        self.loss = loss_registry.parse(loss)

    def call(self, inputs, training=False, **kwargs):
        return self.output_activation(self.output_layer(inputs))

    def compute_output_shape(self, input_shape):
        return self.output_layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(
            self,
            config,
            {"output_layer": tf.keras.layers.serialize, "loss": tf.keras.losses.serialize},
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)
        config = maybe_deserialize_keras_objects(
            config, ["output_layer"], tf.keras.layers.deserialize
        )

        return super().from_config(config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalPrediction(Block):
    """Block that predicts a categorical feature. num_classes is inferred from the"""

    def __init__(
        self,
        feature: Union[str, Tags] = Tags.ITEM_ID,
        weight_tying: bool = False,
        use_bias=True,
        bias_initializer="zeros",
        activation=None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super(CategoricalPrediction, self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic)
        self.bias_initializer = bias_initializer
        self.feature = feature
        self.weight_tying = weight_tying
        self.use_bias = use_bias
        if not self.weight_tying:
            self.output_classes = Dense(
                units=1,  # Will be overwritten in the build-method when we know the number of classes
                # units=self.num_classes,
                bias_initializer=bias_initializer,
                name=f"{self.feature}-prediction",
                activation="linear",
                **kwargs,
            )

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        self.output_activation = tf.keras.layers.Activation(
            activation, dtype="float32", name="predictions"
        )

    @property
    def feature_name(self):
        if isinstance(self.feature, Tags):
            feature_schema = self.context.schema.select_by_tag(self.feature)
            if not len(feature_schema) == 1:
                raise ValueError(
                    f"{self.feature} is a tag, but the schema contains multiple features with that tag."
                )
            return feature_schema.column_names[0]

        return self.feature

    @property
    def num_classes(self):
        return categorical_cardinalities(self.context.schema)[self.feature_name]

    def build(self, input_shape):
        if self.weight_tying:
            self.kernel = self.context.get_embedding(self.feature_name)
            self.bias = self.add_weight(
                name="output_layer_bias",
                shape=(self.num_classes,),
                initializer=self.bias_initializer,
            )
        else:
            self.output_classes.units = self.num_classes
            self.output_classes.build(input_shape)
            self.kernel = self.output_classes.kernel
            self.bias = self.output_classes.bias
        return super().build(input_shape)

    def embedding_lookup(self, inputs, **kwargs):
        kernel = self.kernel if self.weight_tying else tf.transpose(self.kernel)

        return embedding_ops.embedding_lookup(kernel, inputs, **kwargs)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        if not self.weight_tying:
            return self.output_classes(inputs, training=training)

        kernel = tf.transpose(self.kernel)

        # This code is the same as in Dense
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, sparse_tensor.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    kernel, ids, weights, combiner='sum')
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.output_activation is not None:
            outputs = self.output_activation(outputs)

        return outputs

    def apply_mask(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
        # targets = self.context[self.item_id_feature_name]
        targets = self.prediction_block.get_targets(outputs)
        mask = self.context.get_mask()
        targets = tf.where(mask, targets, self.context.padding_idx)

        outputs = remove_pad_3d(outputs.copy_with_updates(targets=targets))

        # Convert labels to one-hot if necessary
        if outputs.targets.shape != outputs.predictions.shape:
            num_classes = tf.shape(outputs.predictions)[-1]
            targets = transform_label_to_onehot(outputs.targets, num_classes)
            outputs = outputs.copy_with_updates(targets=targets)

        return outputs

    def get_targets(
            self,
            inputs: Union[TabularData, tf.Tensor],
            features: DictWithSchema,
            targets: Optional[Union[TabularData, tf.Tensor]] = None,
            **kwargs
    ):
        if not targets:
            targets = features[self.feature_name]

        return targets

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_classes,)


class DotProduct(Layer):
    def call(self, inputs, training=False, testing=False):
        scores = tf.reduce_sum(
            tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
        )

        self.targets = inputs["item_id"]

        return scores

    def get_targets(
            self,
            inputs: Union[TabularData, tf.Tensor],
            features: DictWithSchema,
            targets: Optional[Union[TabularData, tf.Tensor]] = None,
            **kwargs
    ):
        if targets is None:
            if isinstance(inputs, dict) and "item_id" in inputs:
                targets = inputs["item_id"]
            else:
                targets = features[Tags.ITEM_ID]

        return targets


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MultiClassClassificationTask(PredictionTask):
    """
    Prediction task for multi-class classification.

    Parameters
    ----------
    target_name : Optional[str], optional
        Label name, by default None
    task_name: str, optional
        The name of the task.
    task_block: Block, optional
        The block to use for the task.
    loss: LossType, optional
        The loss to use for the task.
        Defaults to "sparse_categorical_crossentropy".
    metrics: MetricOrMetrics, optional
        The metrics to use for the task. Defaults to [accuracy].
    """

    # DEFAULT_LOSS = "sparse_categorical_crossentropy"
    DEFAULT_LOSS = "categorical_crossentropy"
    SUPPORTED_LOSSES = [
        "sparse_categorical_crossentropy",
        "categorical_crossentropy"
    ]
    DEFAULT_METRICS: MetricOrMetrics = (tf.keras.metrics.Accuracy,)

    def __init__(
        self,
        prediction_block: CategoricalPrediction,
        target_name: Optional[Union[str, Schema]] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        logits_temperature: float = 1.0,
        **kwargs,
    ):
        if logits_temperature != 1:
            logits_scaler = LogitsTemperatureScaler(logits_temperature)
            post = post.connect(logits_scaler) if post else logits_scaler

        super().__init__(
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            post=post,
            **kwargs,
        )
        self.prediction_block = prediction_block

    # @classmethod
    # def from_schema(
    #     cls,
    #     schema: Schema,
    #     feature: str = Tags.ITEM_ID,
    #     weight_tying: bool = False,
    #     masking: bool = False,
    #     logits_temperature: float = 1.0,
    #     bias_initializer="zeros",
    #     kernel_initializer="random_normal",
    #     pre: Optional[Block] = None,
    #     post: Optional[Block] = None,
    #     **kwargs,
    # ) -> "MultiClassClassificationTask":
    #     """Create from Schema."""
    #     block = CategoricalPrediction(
    #         schema,
    #         feature_name,
    #         bias_initializer=bias_initializer,
    #         kernel_initializer=kernel_initializer,
    #     )
    #
    #     return cls(
    #         block,
    #         pre=pre,
    #         post=post,
    #         weight_tying=weight_tying,
    #         masking=masking,
    #         logits_temperature=logits_temperature,
    #         **kwargs,
    #     )

    def to_contrastive(
            self,
            *samplers: ItemSampler,
            item_metadata_schema: Optional[Schema] = None,
            item_id_tag: Tags = Tags.ITEM_ID,
            query_id_tag: Tags = Tags.USER_ID,
            downscore_false_negatives: bool = True,
            **kwargs
    ) -> ContrastiveLearningTask:
        from merlin.models.tf import ContrastiveLearningTask

        return ContrastiveLearningTask(
            *samplers,
            prediction_block=self.prediction_block,
            item_metadata_schema=item_metadata_schema,
            item_id_tag=item_id_tag,
            query_id_tag=query_id_tag,
            downscore_false_negatives=downscore_false_negatives,
            pre=self.pre,
            post=self.post,
            **kwargs,
        )

    def to_sampled_softmax(self, num_samples: int = 100, **kwargs) -> ContrastiveLearningTask:
        return self.to_contrastive(PopularityBasedSampler(max_num_samples=num_samples), **kwargs)

    """
    # Ranking model
    model = DLRMModel(schema, ..., prediction_tasks=ItemPrediction(schema).to_contrastive("popularity"))
    
    # Two-tower model
    prediction_task = ContrastiveLearningTask(schema, "in-batch")
    # OR:
    prediction_task = ItemPrediction(schema, dot_product=True).to_contrastive("in-batch")
    
    # YoutubeDNN
    model = MLPBlock(...).to_model(
        schema, 
        prediction_task=ItemPrediction(schema, weight_tying=False).to_contrastive("in-batch")
    )
    
    model.compile(prediction_task=prediction_task)
    model.fit(train_ds)
    
    # Set up contrastive learning as context-manager
    prediction_task = ItemPrediction(schema, dot_product=True)
    with prediction_task.to_contrastive(samplers="in-batch"):
        model.fit(train_ds)
        model.evaluate(eval_ds)
    
    """

    def call(
            self,
            inputs: Union[TabularData, tf.Tensor],
            features: DictWithSchema,
            targets=None,
            training=False,
            testing=False
    ) -> Union[tf.Tensor, TaskResults]:
        predictions = self.prediction_block(inputs, training=training, testing=testing)

        if training or testing:
            targets = self.prediction_block.get_targets(
                inputs, features, targets, training=training, testing=testing
            )

            return TaskResults(predictions=predictions, targets=targets)

    # def pre_loss(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
    #     if self.context.has_mask:
    #         outputs = self.prediction_block.apply_mask(outputs, **kwargs)
    #
    #     return outputs

    def get_targets(
            self,
            inputs: Union[TabularData, tf.Tensor],
            features: DictWithSchema,
            targets: Optional[Union[TabularData, tf.Tensor]] = None,
            **kwargs
    ):
        targets = self.prediction_block.get_targets(inputs, features, targets, **kwargs)

        if self.context.has_mask:
            mask = self.context.get_mask()
            targets = tf.where(mask, targets, self.context.padding_idx)
            targets = remove_pad_3d_targets(targets)

        return targets

    def compute_output_shape(self, input_shape):
        return self.prediction_block.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, {"loss": tf.keras.losses.serialize})

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)

        return super().from_config(config)


def ItemPredictionTask(
        schema: Schema,
        dot_product: bool = False,
        weight_tying: bool = True,
        feature: Union[str, Tags] = Tags.ITEM_ID,
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        normalize: bool = True,
        **kwargs
) -> MultiClassClassificationTask:
    if dot_product:
        masking = False
        if normalize:
            pre = L2Norm().connect(pre) if pre else L2Norm()
        prediction_block = DotProduct(**kwargs)
    else:
        prediction_block = CategoricalPrediction(
            feature, weight_tying=weight_tying, **kwargs
        )

    return MultiClassClassificationTask(
        prediction_block,
        pre=pre,
        post=post,
        target_name=target_name,
        task_name=task_name,
        task_block=task_block,
        logits_temperature=logits_temperature,
    )
