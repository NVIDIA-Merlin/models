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
from typing import Tuple

import tensorflow as tf
from merlin.schema import Schema, Tags

from ...utils.schema import categorical_cardinalities
from ..core import Block


@Block.registry.register_with_multiple_names("sampling-bias-correction")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
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

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        embedding_table = self.context.get_embedding(self.item_id_feature_name)
        logits = tf.matmul(inputs, embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits


@Block.registry.register_with_multiple_names("remove_pad_3d")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
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

    def compute_output_shape(self, input_shape):
        return input_shape

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
@tf.keras.utils.register_keras_serializable(package="merlin_models")
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


# TODO: Implement this for the MIND prediction: https://arxiv.org/pdf/1904.08030.pdf
class LabelAwareAttention(Block):
    def predict(
        self, predictions, targets=None, training=False, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("TODO")
