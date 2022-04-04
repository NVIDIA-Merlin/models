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

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.ops import array_ops

from merlin.models.tf.blocks.core.base import Block, PredictionOutput
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.registry import Registry
from merlin.schema import Tags

masking_registry = Registry("tf.masking")

MASK_SEQUENCE_PARAMETERS_DOCSTRING = """
    padding_idx: int
        Index of padding item, used for masking and for getting batch of sequences
        with the same length.
        Defaults to 0
    eval_on_last_item_seq_only: bool
        When set to True, predict only the last non-padded item during evaluation
        Defaults to True
    item_id_feature_name: str
        Name of the column containing the item ids
        Defaults to `item_id`
"""


@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
@Block.registry.register_with_multiple_names("masking_block")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MaskingBlock(Block):
    """Base class to prepare masked items inputs/labels for item prediction task.
    The masking schema sets the items to be predicted (labels) and masks (hides)
    their positions in the sequence so that they are not used by the model
    for prediction.

    We currently provide 2 different masking schemes:
        - Causal LM (clm)
        - Masked LM (mlm)

    This class can be extended to add custom masking scheme.

    Parameters:
    ----------
        {mask_sequence_parameters}

    Returns:
    -------
        Transformed inputs where masked positions are replaced by
        a trainable mask embedding.
    """

    def __init__(
        self,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        item_id_feature_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.padding_idx = padding_idx
        self.eval_on_last_item_seq_only = eval_on_last_item_seq_only
        self.item_id_feature_name = item_id_feature_name

    def build(self, input_shapes):
        self.context.add_variable(
            tf.Variable(
                initial_value=tf.zeros([1, input_shapes[1]], dtype=tf.bool),
                name="masking_schema",
                trainable=False,
                validate_shape=False,
                shape=tf.TensorShape([None, input_shapes[1]]),
            )
        )

        self.masked_item_embedding = self.add_weight(
            name="mask_embedding",
            trainable=True,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
            shape=[input_shapes[-1]],
            dtype=tf.float32,
        )

        self.label_seq_trg_eval = tf.Variable(
            tf.zeros(shape=[1, input_shapes[1]], dtype=tf.int64),
            name="target_labels",
            dtype=tf.int64,
            trainable=False,
            shape=tf.TensorShape([None, input_shapes[1]]),
        )

        super().build(input_shapes)

    def add_features_to_context(self, feature_shapes) -> List[str]:
        f = self.item_id_feature_name or self.schema.select_by_tag(Tags.ITEM_ID).column_names[0]

        return [f]

    def compute_mask_schema(self, items: tf.Tensor, training: bool = False) -> tf.Tensor:
        raise NotImplementedError()

    def apply_mask_to_inputs(self, inputs: tf.Tensor, schema: tf.Tensor) -> tf.Tensor:
        inputs = tf.where(
            tf.cast(tf.expand_dims(schema, -1), tf.bool),
            inputs,
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
        )
        return inputs

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        items = self.context[self.schema.select_by_tag(Tags.ITEM_ID)]
        mask_schema = self.compute_mask_schema(items, training=training)
        inputs = self.apply_mask_to_inputs(inputs, mask_schema)
        return inputs

    def get_config(self):
        config = super(MaskingBlock, self).get_config()
        config.update(
            {
                "padding_idx": self.padding_idx,
                "eval_on_last_item_seq_only": self.eval_on_last_item_seq_only,
            }
        )
        return config


@masking_registry.register_with_multiple_names("clm", "causal")
@docstring_parameter(mask_sequence_parameters=MASK_SEQUENCE_PARAMETERS_DOCSTRING)
@Block.registry.register_with_multiple_names("causal_language_modeling")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class CausalLanguageModeling(MaskingBlock):
    """
    In Causal Language Modeling (clm) you predict the next item based on past positions of the
    sequence. Future positions are masked.
    Parameters
    ----------
    {mask_sequence_parameters}
    train_on_last_item_seq_only: Optional[bool]
        predict only the last item during training.
        Defaults to True.
    """

    def __init__(
        self,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        train_on_last_item_seq_only: bool = True,
        **kwargs
    ):
        super(CausalLanguageModeling, self).__init__(
            padding_idx=padding_idx, eval_on_last_item_seq_only=eval_on_last_item_seq_only, **kwargs
        )
        self.train_on_last_item_seq_only = train_on_last_item_seq_only

    def compute_mask_schema(self, items: tf.Tensor, training: bool = False) -> tf.Tensor:
        items = tf.cast(tf.squeeze(items), tf.int64)
        if (self.eval_on_last_item_seq_only and not training) or (
            self.train_on_last_item_seq_only and training
        ):
            mask_labels = items != self.padding_idx
            last_item_sessions = tf.reduce_sum(tf.cast(mask_labels, items.dtype), axis=1) - 1

            rows_ids = tf.range(tf.shape(items)[0], dtype=items.dtype)
            self.label_seq_trg_eval.assign(tf.zeros(tf.shape(items), dtype=tf.int64))

            indices = tf.concat(
                [tf.expand_dims(rows_ids, 1), tf.expand_dims(last_item_sessions, 1)], axis=1
            )
            self.label_seq_trg_eval.scatter_nd_update(
                indices=indices, updates=tf.gather_nd(items, indices)
            )
            # Updating labels and mask
            mask_labels = self.label_seq_trg_eval != self.padding_idx

        else:
            labels = items[:, 1:]
            # pad shifted sequence to original length
            labels = tf.concat(
                [labels, tf.zeros((tf.shape(items)[0], 1), dtype=labels.dtype)],
                axis=-1,
            )
            mask_labels = labels != self.padding_idx

        # store boolean tensor related to masked targets
        self.context["masking_schema"].assign(mask_labels)

        return mask_labels

    def apply_mask_to_inputs(self, inputs: tf.Tensor, mask_schema: tf.Tensor) -> tf.Tensor:
        pos_emb_inp = inputs[:, :-1]
        # Adding a masked item in the sequence to return to the initial sequence length.
        pos_emb_inp = tf.concat(
            [
                pos_emb_inp,
                tf.zeros(
                    (tf.shape(pos_emb_inp)[0], 1, pos_emb_inp.shape[2]), dtype=pos_emb_inp.dtype
                ),
            ],
            axis=1,
        )

        pos_emb_inp = tf.where(
            tf.cast(tf.expand_dims(mask_schema, -1), tf.bool),
            pos_emb_inp,
            tf.cast(self.masked_item_embedding, dtype=inputs.dtype),
        )
        return pos_emb_inp


@masking_registry.register_with_multiple_names("mlm", "masked")
@Block.registry.register_with_multiple_names("masked_language_modeling")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MaskedLanguageModeling(MaskingBlock):
    """
    In Masked Language Modeling (mlm) you randomly select some positions of the sequence to be
    predicted, which are masked.
    During training, the Transformer layer is allowed to use positions on the right (future info).
    During inference, all past items are visible for the Transformer layer, which tries to predict
    the next item.
    Parameters
    ----------
    {mask_sequence_parameters}
    mlm_probability: Optional[float]
        Probability of an item to be selected (masked) as a label of the given sequence.
        p.s. We enforce that at least one item is masked for each sequence, so that the network can
        learn something with it.
        Defaults to 0.15
    """

    def __init__(
        self,
        padding_idx: int = 0,
        eval_on_last_item_seq_only: bool = True,
        mlm_probability: float = 0.15,
        **kwargs
    ):
        super(MaskedLanguageModeling, self).__init__(
            padding_idx=padding_idx, eval_on_last_item_seq_only=eval_on_last_item_seq_only, **kwargs
        )
        self.mlm_probability = mlm_probability
        self.labels = tf.Variable(
            tf.zeros(shape=[1, 1], dtype=tf.int64),
            name="target_labels",
            dtype=tf.int64,
            trainable=False,
            shape=tf.TensorShape([None, None]),
        )

    def get_config(self):
        config = super(MaskedLanguageModeling, self).get_config()
        config.update(
            {
                "mlm_probability": self.mlm_probability,
            }
        )
        return config

    def compute_mask_schema(self, item_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Compute the mask schema for masked language modeling task
        the function is based on HuggingFace's transformers/data/data_collator.py
        """
        item_ids = tf.squeeze(item_ids)
        item_ids = tf.cast(item_ids, dtype=tf.int64)
        self.labels.assign(tf.cast(tf.fill(tf.shape(item_ids), self.padding_idx), tf.int64))
        non_padded_mask = tf.cast(item_ids != self.padding_idx, self.labels.dtype)
        rows_ids = tf.range(tf.shape(item_ids)[0], dtype=tf.int64)

        # During training, masks labels to be predicted according to a probability, ensuring that
        # each session has at least one label to predict
        if training:
            probability_matrix = tf.cast(
                backend.random_bernoulli(array_ops.shape(item_ids), p=self.mlm_probability),
                self.labels.dtype,
            )
            mask_labels = probability_matrix * non_padded_mask
            self.labels.assign(
                tf.where(
                    tf.cast(mask_labels, tf.bool),
                    item_ids,
                    tf.cast(tf.fill(tf.shape(item_ids), self.padding_idx), dtype=item_ids.dtype),
                )
            )

            # Set at least one item in the sequence to mask, so that the network
            # can learn something with this session
            one_random_index_by_session = tf.random.categorical(
                tf.math.log(tf.cast(non_padded_mask, tf.float32)), num_samples=1
            )
            indices = tf.concat([tf.expand_dims(rows_ids, 1), one_random_index_by_session], axis=1)
            self.labels.scatter_nd_update(indices=indices, updates=tf.gather_nd(item_ids, indices))
            mask_labels = tf.cast(self.labels != self.padding_idx, self.labels.dtype)

            # If a sequence has only masked labels, unmask one of the labels
            sequences_with_only_labels = tf.reduce_sum(mask_labels, axis=1) == tf.reduce_sum(
                non_padded_mask, axis=1
            )
            sampled_labels_to_unmask = tf.random.categorical(
                tf.math.log(tf.cast(mask_labels, tf.float32)), num_samples=1
            )

            labels_to_unmask = tf.boolean_mask(sampled_labels_to_unmask, sequences_with_only_labels)
            rows_to_unmask = tf.boolean_mask(rows_ids, sequences_with_only_labels)
            indices = tf.concat([tf.expand_dims(rows_to_unmask, 1), labels_to_unmask], axis=1)
            num_updates = tf.shape(indices)[0]
            self.labels.scatter_nd_update(
                indices, tf.cast(tf.fill((num_updates,), self.padding_idx), self.labels.dtype)
            )
            mask_labels = self.labels != self.padding_idx

        elif self.eval_on_last_item_seq_only:
            last_item_sessions = tf.reduce_sum(non_padded_mask, axis=1) - 1

            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(tf.expand_dims(last_item_sessions, 1), tf.int64),
                ],
                axis=1,
            )
            self.labels.scatter_nd_update(indices=indices, updates=tf.gather_nd(item_ids, indices))
            mask_labels = self.labels != self.padding_idx
        else:
            labels = item_ids[:, 1:]
            labels = tf.concat(
                [
                    labels,
                    tf.zeros((tf.shape(labels)[0], 1), dtype=labels.dtype),
                ],
                axis=-1,
            )
            mask_labels = labels != self.padding_idx

        # Store boolean tensor related to masked targets
        self.context["masking_schema"].assign(mask_labels)
        return mask_labels


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

    def call_outputs(
        self, outputs: PredictionOutput, training: bool = True, **kwargs
    ) -> "PredictionOutput":
        targets = self.context[self.item_id_feature_name]
        mask = self.context.get_mask()
        targets = tf.where(mask, targets, self.padding_idx)
        return outputs.copy_with_updates(
            targets=targets,
        )
