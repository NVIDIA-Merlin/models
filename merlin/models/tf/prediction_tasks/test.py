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

from typing import Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.layers import Dense
from tensorflow.python.ops import embedding_ops, math_ops, sparse_ops, gen_math_ops, standard_ops, nn_ops

from merlin.models.tf.blocks.core.base import Block, MetricOrMetrics, PredictionOutput
from merlin.models.tf.blocks.core.masking import MaskingHead
from merlin.models.tf.blocks.core.transformations import LogitsTemperatureScaler, RemovePad3D, L2Norm, remove_pad_3d
from merlin.models.tf.prediction_tasks.base import PredictionTask
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.models.utils.schema_utils import categorical_cardinalities
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CategoricalPrediction(Block):
    """Block that predicts a categorical feature. num_classes is inferred from the"""

    def __init__(
        self,
        schema: Schema,
        feature_name: Optional[str] = None,
        weight_tying: bool = False,
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
        self.feature_name = feature_name or schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.num_classes = categorical_cardinalities(schema)[self.feature_name]
        self.weight_tying = weight_tying
        if not self.weight_tying:
            self.output_classes = Dense(
                units=self.num_classes,
                bias_initializer=bias_initializer,
                name=f"{self.feature_name}-prediction",
                activation="linear",
                **kwargs,
            )

        # To ensure that the output is always fp32, avoiding numerical
        # instabilities with mixed_float16 policy
        self.output_activation = tf.keras.layers.Activation(
            activation, dtype="float32", name="predictions"
        )

    def build(self, input_shape):
        if self.weight_tying:
            self.kernel = self.context.get_embedding(self.feature_name)
            self.bias = self.add_weight(
                name="output_layer_bias",
                shape=(self.num_classes,),
                initializer=self.bias_initializer,
            )
        else:
            self.output_classes.build(input_shape)
            self.kernel = self.output_classes.kernel
            self.bias = self.output_classes.bias
        return super().build(input_shape)

    def embedding_lookup(self, inputs, **kwargs):
        return embedding_ops.embedding_lookup(self.kernel, inputs, **kwargs)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
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
                    self.kernel, ids, weights, combiner='sum')
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_classes,)


class DotProduct(Layer):
    def call(self, inputs, training=False, testing=False):
        if isinstance(inputs, dict) and not (training or testing):
            scores = tf.reduce_sum(
                tf.multiply(inputs["query"], inputs["item"]), keepdims=True, axis=-1
            )
            return scores

        return inputs


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MultiClassClassificationTaskNew(PredictionTask):
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

    DEFAULT_LOSS = "sparse_categorical_crossentropy"
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

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        feature_name: str = Tags.ITEM_ID,
        weight_tying: bool = False,
        masking: bool = False,
        logits_temperature: float = 1.0,
        bias_initializer="zeros",
        kernel_initializer="random_normal",
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        **kwargs,
    ) -> "MultiClassClassificationTaskNew":
        """Create from Schema."""
        block = CategoricalPrediction(
            schema,
            feature_name,
            bias_initializer=bias_initializer,
            kernel_initializer=kernel_initializer,
        )

        return cls(
            block,
            pre=pre,
            post=post,
            weight_tying=weight_tying,
            masking=masking,
            logits_temperature=logits_temperature,
            **kwargs,
        )

    def call(self, inputs, training=False, **kwargs):
        return self.prediction_block(inputs, training=training, **kwargs)

    def pre_loss(self, outputs: PredictionOutput, **kwargs) -> "PredictionOutput":
        if self.context.has_mask:
            targets = self.context[self.item_id_feature_name]
            mask = self.context.get_mask()
            targets = tf.where(mask, targets, self.padding_idx)

            outputs = remove_pad_3d(outputs.copy_with_updates(targets=targets))

            return outputs

        return outputs

    def get_config(self):
        config = super().get_config()
        config = maybe_serialize_keras_objects(self, config, {"loss": tf.keras.losses.serialize})

        return config

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["loss"], tf.keras.losses.deserialize)

        return super().from_config(config)


def ItemPredictionTask(
        schema,
        dot_product: bool = False,
        weight_tying: bool = True,
        pre: Optional[Block] = None,
        post: Optional[Block] = None,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        logits_temperature: float = 1.0,
        normalize: bool = True,
        **kwargs
):
    if dot_product:
        masking = False
        if normalize:
            pre = L2Norm().connect(pre) if pre else L2Norm()
        prediction_block = DotProduct(**kwargs)
    else:
        prediction_block = CategoricalPrediction(
            schema, weight_tying=weight_tying, **kwargs
        )

    return MultiClassClassificationTaskNew(
        prediction_block,
        weight_tying=weight_tying,
        pre=pre,
        post=post,
        target_name=target_name,
        task_name=task_name,
        task_block=task_block,
        logits_temperature=logits_temperature,
    )
