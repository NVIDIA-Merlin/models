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

import tensorflow as tf
from tensorflow.keras.backend import random_bernoulli

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.typing import TabularData
from merlin.models.utils.schema_utils import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceRandomTargetMasking(Block):
    """This block implements the Masked Language Modeling (MLM) training approach
    introduced in BERT (NLP) and later adapted to RecSys by BERT4Rec [1].
    Given an input tf.RaggedTensor with sequences of embeddings
    and the corresponding sequence of item ids, some positions are randomly selected (masked)
    to be the targets for prediction.
    The input embeddings at the masked positions are
    replaced by a common trainable embedding, to avoid leakage of the targets information.
    The targets are output being the same as the input id sequence. The masked targets
    can be discovered later by checking the outputs._keras_mask.
    This transformation is only applied during training, as during inference you want
    to use all available information of the sequence for prediction.

    References
    ----------
    .. [1] Sun, Fei, et al. "BERT4Rec: Sequential recommendation with bidirectional encoder
           representations from transformer." Proceedings of the 28th ACM international
           conference on information and knowledge management. 2019.

    Parameters
    ----------
    schema : Schema
        The input schema, that will be used to discover the name
        of the item id column
    masking_prob : float, optional
        Probability of an item to be selected (masked) as a label of the given sequence.
        Note: We enforce that at least one item is masked for each sequence, so that it
        is useful for training, by default 0.2
    """

    def __init__(self, schema: Schema, masking_prob: float = 0.2, **kwargs):
        self.schema = schema
        self.item_id_col = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.masking_prob = masking_prob
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "The inputs must be a 3D tf.RaggedTensor (batch_size, seq_length, vector_dim)"
            )
        self.hidden_size = input_shape[-1]
        if self.hidden_size is None:
            raise ValueError("The tf.RaggedTensor last dim cannot be None")
        # Create a trainable embedding to replace masked interactions
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        self.masked_item_embedding = tf.Variable(
            initializer(shape=[self.hidden_size], dtype=tf.float32)
        )

        return super().build(input_shape)

    def call(
        self,
        inputs: tf.RaggedTensor,
        features: TabularData = None,
        training: bool = False,
    ) -> Prediction:
        """Masks some items from the input sequence to be the targets
        and output the input tensor with replaced embeddings for masked
        elements and also the targets (copy of the items ids sequence)

        Parameters
        ----------
        inputs : tf.RaggedTensor
            A ragged tensor with sequences of vectors.
            Needs to be 3D (batch_size, sequence_length, vectors_dim)
        features : TabularData, optional
            The input features, so that the item id column values
            can be extracted, by default None
        training : bool, optional
            A flag indicating whether model is being trained or not, by default False

        Returns
        -------
        Prediction
            Returns a Prediction(replaced_inputs, targets)
        """
        if not isinstance(inputs, tf.RaggedTensor):
            raise ValueError(
                "Only a tf.RaggedTensor is accepted to "
                "represent the input sequences.\n "
                "If you have a tf.Tensor and a padding mask "
                "you can use tf.ragged.boolean_mask(tensor, padding_mask) "
                "to convert it to a tf.RaggedTensor"
            )
        if len(inputs.shape.as_list()) != 3:
            raise ValueError(
                "The inputs must be a 3D tf.RaggedTensor (batch_size, seq_length, vector_dim)"
            )

        outputs = inputs
        targets = None
        self.target_mask = None
        # During training, masks labels to be predicted according to a probability, ensuring that
        # each session has at least one label to predict
        if training:
            if features is None or self.item_id_col not in features:
                raise ValueError(
                    f"The features provided does contain the item id "
                    f"column ({self.item_id_col})"
                )
            item_id_seq = features[self.item_id_col]
            self.target_mask = self._generate_target_mask(item_id_seq)
            outputs = self._mask_inputs(inputs)
            targets = features[self.item_id_col]
        return Prediction(outputs, targets)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor):
        batch_size = tf.shape(ids_seq)[0]
        row_lengths = ids_seq.row_lengths(1)

        assertion_min_seq_length = tf.Assert(tf.reduce_all(row_lengths > 1), [row_lengths])

        with tf.control_dependencies([assertion_min_seq_length]):
            # Targets are masked according to a probability
            target_mask_by_prob = self._get_masked_by_prob(
                batch_size, row_lengths, prob=self.masking_prob
            )
            # Exactly one target is masked per row
            one_target_mask = self._get_one_masked(batch_size, row_lengths)

            # For sequences (rows) with either all or none elements sampled (masked) as targets
            # as those sequences would be invalid for training
            # the row mask is replaced by a row mask that contains exactly one masked target
            replacement_cond = tf.logical_or(
                tf.logical_not(tf.reduce_any(target_mask_by_prob, axis=1)),
                tf.reduce_all(target_mask_by_prob, axis=1),
            )
            target_mask = tf.where(
                tf.expand_dims(replacement_cond, -1), one_target_mask, target_mask_by_prob
            )
            padding_mask = tf.sequence_mask(row_lengths)
            target_mask_ragged = tf.ragged.boolean_mask(target_mask, padding_mask)

            return target_mask_ragged

    @staticmethod
    def _get_masked_by_prob(
        batch_size: tf.Tensor, row_lengths: tf.Tensor, prob: float
    ) -> tf.Tensor:
        """Generates a dense mask boolean tensor with True values
        for randomly selected targets
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        output = tf.cast(random_bernoulli([batch_size, max_seq_length], p=prob), tf.bool)
        padding_mask = tf.sequence_mask(row_lengths)
        # Ignoring masked items in the padding positions
        output = tf.logical_and(output, padding_mask)
        return output

    @staticmethod
    def _get_one_masked(batch_size: tf.Tensor, row_lengths: tf.Tensor):
        """Generates a dense mask boolean tensor where for each tensor (row)
        there is exactly one True value (selected target)
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        random_targets_indices = tf.cast(
            tf.math.floor(
                (
                    tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0, dtype=tf.float32)
                    * tf.cast(row_lengths, tf.float32)
                )
            ),
            tf.int32,
        )

        one_target_mask = tf.cast(tf.one_hot(random_targets_indices, max_seq_length), tf.bool)
        return one_target_mask

    def _mask_inputs(self, inputs: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        Replaces in the input tensor the values masked as targets by a common trainable
        embedding
        """
        output = tf.where(
            tf.cast(tf.expand_dims(self.target_mask, -1), tf.bool),
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
            inputs,
        )
        return output

    def compute_mask(self, inputs, mask=None):
        """Is called by Keras and returns the targets mask that will
        be assigned to the input tensor, being accessible
        by inputs._keras_mask
        """
        return self.target_mask

    def get_config(self):
        config = super().get_config()
        config["schema"] = schema_to_tensorflow_metadata_json(self.schema)
        config["masking_prob"] = self.masking_prob
        return config

    @classmethod
    def from_config(cls, config):
        # block = tf.keras.utils.deserialize_keras_object(config.pop("block"))
        schema = tensorflow_metadata_json_to_schema(config.pop("schema"))
        masking_prob = config.pop("masking_prob")
        return cls(schema, masking_prob, **config)
