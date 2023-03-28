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
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.backend import random_bernoulli

from merlin.models.tf.core import combinators
from merlin.models.tf.core.base import Block, BlockType, PredictionOutput
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.transforms.features import PrepareFeatures
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.dependencies import is_transformers_available
from merlin.schema import ColumnSchema, Schema, Tags


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
            If the predictions are 3-D vectors (sequential task),
            flatten the predictions vectors to keep only the ones related to target positions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        targets = tf.reshape(targets, (-1,))
        non_pad_mask = targets != self.padding_idx
        targets = tf.boolean_mask(targets, non_pad_mask)

        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"

        if len(tuple(predictions.get_shape())) == 3:
            predictions = tf.reshape(predictions, (-1, predictions.shape[-1]))
            predictions = tf.boolean_mask(
                predictions, tf.broadcast_to(tf.expand_dims(non_pad_mask, 1), tf.shape(predictions))
            )

        return outputs.copy_with_updates(predictions=predictions, targets=targets)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequenceTransform(TabularBlock):
    """Base class for preparing sequential inputs and targets.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        P.s. The PrepareFeatures() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
    transformer:  Optional[TransformerBlock]
        The transformer block that leverages the group of sequences returned
        by the given SequenceTransform, by default None.
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        pre: Optional[BlockType] = None,
        transformer=None,
        **kwargs,
    ):
        _pre = PrepareFeatures(schema)
        if pre:
            _pre = _pre.connect(pre)
        super().__init__(pre=_pre, schema=schema, **kwargs)

        self.target = target
        self.target_name = self._get_target(target)
        self.transformer = transformer

    def _get_target(self, target):
        if (
            (isinstance(target, str) and target not in self.schema.column_names)
            or (isinstance(target, Tags) and len(self.schema.select_by_tag(target)) > 0)
            or (isinstance(target, ColumnSchema) and target not in self.schema)
        ):
            raise ValueError("The target column needs to be part of the sequential schema")

        target_name = target
        if isinstance(target, ColumnSchema):
            target_name = target.name
        if isinstance(target, Tags):
            if len(self.schema.select_by_tag(target)) > 1:
                raise ValueError(
                    "Only 1 column should the Tag ({target}) provided for target, but"
                    f"the following columns have that tag: "
                    f"{self.schema.select_by_tag(target).column_names}"
                )
            target_name = self.schema.select_by_tag(target).column_names[0]
        return target_name

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        raise NotImplementedError()

    def _check_seq_inputs_targets(self, inputs: TabularData):
        if self.target_name not in inputs:
            raise ValueError(
                f"The inputs provided does contain the target column ({self.target_name})"
            )

        target_shape = inputs[self.target_name].get_shape().as_list()
        if len(target_shape) < 2:
            raise ValueError(
                f"The sequential target column ({self.target_name}) cannot be a 1D tensor,"
                f" but the shape is {target_shape}"
            )
        if target_shape[1] == 1:
            raise ValueError(
                f"The 2nd dim of the target column ({self.target_name}) should be greater"
                " than 1, so that the sequential input can be shifted as target"
            )

        seq_inputs_shapes = {
            col: inputs[col].get_shape().as_list() for col in self.schema.column_names
        }

        seq_shapes = list(seq_inputs_shapes.values())
        if not all(x == seq_shapes[0] for x in seq_shapes):
            raise ValueError(
                "The sequential inputs must have the same shape, but the shapes "
                f"are different: {seq_inputs_shapes}"
            )

    def compute_output_shape(self, input_shape):
        new_input_shapes = dict()
        for k, v in input_shape.items():
            new_input_shapes[k] = v
            if k in self.schema.column_names:
                # If it is a list/sparse feature (in tuple representation), uses the offset as shape
                if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                    new_input_shapes[k] = tf.TensorShape([v[1][0], None])
                else:
                    new_input_shapes[k] = v

        return new_input_shapes

    def get_config(self):
        """Returns the config of the layer as a Python dictionary."""
        config = super().get_config()
        config["target"] = self.target

        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config. Returning the instance."""
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])
        schema = config.pop("schema")
        target = config.pop("target")
        return cls(schema, target, **config)

    def configure_for_train(self):
        """Method called by the model.fit() to set additional model's
        configuration before calling keras parent class `fit()`
        """
        pass

    def configure_for_test(self):
        """Method called by the model.evaluate() to check any custom model's
        configuration before calling keras parent class `evaluate()`
        """
        pass


@Block.registry.register_with_multiple_names("seq_predict_next")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequencePredictNext(SequenceTransform):
    """Prepares sequential inputs and targets for next-item prediction.
    The target is extracted from the shifted sequence of item ids and
    the sequential input features are truncated in the last position.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        P.s. The PrepareFeatures() is called automatically before the pre
        to convert the list features representation
    """

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        # Shifts the target column to be the next item of corresponding input column
        new_target = inputs[self.target_name][:, 1:]
        if targets is None:
            targets = dict({self.target_name: new_target})
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            raise ValueError("Targets should be None or a dict of tensors")

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                # Removes the last item of the sequence, as it belongs to the target
                new_inputs[k] = v[:, :-1]
            else:
                new_inputs[k] = v

        return (new_inputs, targets)

    def compute_output_shape(self, input_shape):
        new_input_shapes = dict()
        for k, v in input_shape.items():
            new_input_shapes[k] = v
            if k in self.schema.column_names:
                # If it is a list/sparse feature (in tuple representation), uses the offset as shape
                if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                    new_input_shapes[k] = tf.TensorShape([v[1][0], None])
                else:
                    # Reducing 1 position of the seq length
                    new_input_shapes[k] = tf.TensorShape([v[0], v[1] - 1])

        return new_input_shapes

    def compute_mask(self, inputs, mask=None):
        new_item_id_seq = inputs[self.target_name][:, :-1]
        target_mask = tf.RaggedTensor.from_row_lengths(
            values=tf.ones_like(new_item_id_seq.flat_values, dtype=tf.bool),
            row_lengths=new_item_id_seq.row_lengths(),
        )
        self.target_mask = tf.squeeze(target_mask, axis=-1)

        targets_mask = dict({self.target_name: self.target_mask})
        inputs_mask = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                inputs_mask[k] = self.target_mask
            else:
                inputs_mask[k] = None
        return (inputs_mask, targets_mask)

    def configure_for_train(self):
        """Method called by the model.fit() to set the specialized
        `masking_post` and `masking_pre` needed by the TransformerBlock
        to align with the SequencePredictNext outputs.
        """
        if self.transformer is not None:
            if not is_transformers_available():
                raise ImportError("HuggingFace library `transformers` is required")
            from merlin.models.tf.transformers.transforms import (
                TransformerInferenceHiddenState,
                TransformerOutputToRagged,
            )

            # set the tansformer block with the correct masking block
            self.transformer.masking_post = combinators.SequentialBlock(
                [TransformerOutputToRagged(), TransformerInferenceHiddenState()]
            )
            self.transformer.masking_pre = combinators.SequentialBlock(
                [SequenceCausalLastInference(), ExtractMaskFromTargets()]
            )

    def configure_for_test(self):
        """Method called by the model.evaluate() to check that the
        `masking_post` and `masking_pre` set in the TransformerBlock
        are aligned with the evaluation strategy of SequencePredictNext
        """
        if self.transformer is not None:
            if self.transformer.masking_pre is None:
                raise ValueError(
                    "To evaluate using `SequencePredictNext`, ensure that your TransformerBlock has"
                    " `masking_pre` set as"
                    " `combinators.SequentialBlock("
                    "    [SequenceCausalLastInference(), ExtractMaskFromTargets()]"
                    ")`."
                    " You can automatically set `masking_pre` by passing `SequencePredictNext`"
                    " as the `pre` argument to the `fit` method: "
                    "`model.fit(..., pre=SequencePredictNext(...))`."
                )

            if any(
                isinstance(layer, ReplaceMaskedEmbeddings)
                for layer in self.transformer.masking_pre.layers
            ):
                ValueError(
                    "You cannot use `ReplaceMaskedEmbeddings` as `masking_pre`"
                    " of your TransformerBlock with the `SequencePredictNext`"
                    " evaluation strategy. Please ensure that your Transformer"
                    " model has been trained with `SequencePredictNext`"
                    " by passing it as the `pre` argument to the `fit` method: "
                    "`model.fit(..., pre=SequencePredictNext(...))`."
                )


@Block.registry.register_with_multiple_names("seq_predict_last")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequencePredictLast(SequenceTransform):
    """Prepares sequential inputs and targets for last-item prediction.
    The target is extracted from the last element of sequence of item ids and
    the sequential input features are truncated before the last position.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        P.s. The PrepareFeatures() is called automatically before the pre
        to convert the list features representation
    """

    @tf.function
    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        # Shifts the target column to be the next item of corresponding input column
        new_target = tf.squeeze(inputs[self.target_name][:, -1:], axis=1)
        if targets is None:
            targets = dict({self.target_name: new_target})
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            raise ValueError("Targets should be None or a dict of tensors")

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                # Removes the last item of the sequence, as it belongs to the target
                new_inputs[k] = v[:, :-1]
            else:
                new_inputs[k] = v

        return (new_inputs, targets)

    def compute_output_shape(self, input_shape):
        new_input_shapes = dict()
        for k, v in input_shape.items():
            new_input_shapes[k] = v
            if k in self.schema.column_names:
                # If it is a list/sparse feature (in tuple representation), uses the offset as shape
                if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                    new_input_shapes[k] = tf.TensorShape([v[1][0], None])
                else:
                    # Reducing 1 position of the seq length
                    new_input_shapes[k] = tf.TensorShape([v[0], v[1] - 1])

        return new_input_shapes

    def compute_mask(self, inputs, mask=None):
        new_item_id_seq = inputs[self.target_name][:, :-1]
        self.target_mask = self._generate_target_mask(new_item_id_seq)
        inputs_mask = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                inputs_mask[k] = self.target_mask
            else:
                inputs_mask[k] = None

        return (inputs_mask, self.target_mask)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor) -> tf.RaggedTensor:
        """Returns a bool ragged tensor with the last positions of the sequence masked

        Parameters
        ----------
        ids_seq : tf.RaggedTensor
            Sequence of ids, which are used to infer how many values
            each sequence contains

        Returns
        -------
        tf.RaggedTensor
            Mask tensor, with True at the last positions
        """
        row_lengths = ids_seq.row_lengths(1)
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)

        padding_mask = tf.sequence_mask(row_lengths)
        targets_mask = tf.ragged.boolean_mask(
            tf.cast(tf.one_hot(row_lengths - 1, max_seq_length), tf.bool), padding_mask
        )
        return targets_mask


@Block.registry.register_with_multiple_names("seq_predict_random")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequencePredictRandom(SequenceTransform):
    """Prepares sequential inputs and targets for random-item prediction.
    A random element in the sequence (except the first one) is selected
    as target and all elements before the selected target as used as
    input features.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        P.s. The PrepareFeatures() is called automatically before the pre
        to convert the list features representation
    """

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        batch_size = inputs[self.target_name].nrows()
        seq_length = inputs[self.target_name].row_lengths(1)
        max_length = tf.reduce_max(seq_length)
        random_targets_indices = tf.expand_dims(
            tf.cast(
                tf.math.floor(
                    (
                        tf.random.uniform(
                            shape=[batch_size], minval=0.0, maxval=1.0, dtype=tf.float32
                        )
                        * tf.cast(seq_length - 1, tf.float32)
                    )
                    + 1
                ),
                tf.int32,
            ),
            1,
        )

        positions_matrix = tf.tile(
            tf.expand_dims(tf.range(0, max_length, dtype=tf.int32), 0), [batch_size, 1]
        )
        self.random_mask = positions_matrix < random_targets_indices
        target_mask = positions_matrix == random_targets_indices

        new_target = tf.squeeze(
            tf.ragged.boolean_mask(inputs[self.target_name], target_mask), axis=1
        )
        if targets is None:
            targets = dict({self.target_name: new_target})
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            raise ValueError("Targets should be None or a dict of tensors")

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                new_inputs[k] = tf.ragged.boolean_mask(v, self.random_mask)
            else:
                new_inputs[k] = v

        return (new_inputs, targets)

    def compute_mask(self, inputs, mask=None):
        new_item_id_seq = tf.ragged.boolean_mask(inputs[self.target_name], self.random_mask)

        self.target_mask = self._generate_target_mask(new_item_id_seq)
        inputs_mask = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                inputs_mask[k] = self.target_mask
            else:
                inputs_mask[k] = None

        return (inputs_mask, self.target_mask)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor) -> tf.RaggedTensor:
        """Returns a bool ragged tensor with the last positions of the sequence masked

        Parameters
        ----------
        ids_seq : tf.RaggedTensor
            Sequence of ids, which are used to infer how many values
            each sequence contains

        Returns
        -------
        tf.RaggedTensor
            Mask tensor, with True at the last positions
        """
        row_lengths = ids_seq.row_lengths(1)
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)

        padding_mask = tf.sequence_mask(row_lengths)
        targets_mask = tf.ragged.boolean_mask(
            tf.cast(tf.one_hot(row_lengths - 1, max_seq_length), tf.bool), padding_mask
        )
        return targets_mask


@Block.registry.register_with_multiple_names("seq_target_as_input")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequenceTargetAsInput(SequenceTransform):
    """Creates targets to be equal to one of the sequential input features.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        P.s. The PrepareFeatures() is called automatically before the pre
        to convert the list features representation
    """

    @tf.function
    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        new_target = tf.identity(inputs[self.target_name])
        if targets is None:
            targets = dict({self.target_name: new_target})
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            raise ValueError("Targets should be None or a dict of tensors")

        return (inputs, targets)

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        target = config.pop("target")
        return cls(schema, target, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceMaskRandom(SequenceTargetAsInput):
    """This block implements the Masked Language Modeling (MLM) training approach
    introduced in BERT (NLP) and later adapted to RecSys by BERT4Rec [1].
    Given an input tf.RaggedTensor with sequences of embeddings
    and the corresponding sequence of item ids, some positions are randomly selected (masked)
    to be the targets for prediction.
    The targets are output being the same as the input ids sequence.
    The target masks are returned by using Keras Masking
    (._keras_mask), which is set by the compute_mask() method.

    Note: The SequenceMaskRandom is meant to be used as a `pre` of model.fit(),
    e.g. model.fit(..., pre=SequenceMaskRandom(...)).
    Note: Typically during model.evaluate() you want to evaluate the model
    to predict the last position of the sequence, as it mimics the next-item predicion
    task. In that case, you should use model.evaluate(..., pre=SequenceMaskLast(...))
    instead of SequenceMaskRandom(...).

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
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    masking_prob : float, optional
        Probability of an item to be selected (masked) as a label of the given sequence.
        Note: We enforce that at least one item is masked for each sequence, so that it
        is useful for training, by default 0.2
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        masking_prob: float = 0.2,
        **kwargs,
    ):
        self.masking_prob = masking_prob
        super().__init__(schema, target, **kwargs)

    @tf.function
    def compute_mask(self, inputs, mask=None):
        """Selects (masks) some positions of the targets to be predicted.
        This method is called by Keras after call()
        and returns the targets mask that will
        be assigned to the input tensors and targets, being accessible
        by tensor._keras_mask
        """
        item_id_seq = inputs[self.target_name]
        self.target_mask = self._generate_target_mask(item_id_seq)

        inputs_mask = dict()
        for k in inputs:
            if k in self.schema.column_names:
                inputs_mask[k] = self.target_mask
            else:
                inputs_mask[k] = None

        targets_mask = dict({self.target_name: self.target_mask})
        return (inputs_mask, targets_mask)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor) -> tf.RaggedTensor:
        """Generates a target mask according to the defined probability and
        to the constraints for Masked Language Modeling training (i.e., each
        sequence might have between 1 and length-1 masked positions.)

        Parameters
        ----------
        ids_seq : tf.RaggedTensor
            Sequence of ids, which are used to infer how many values
            each sequence contains

        Returns
        -------
        tf.RaggedTensor
            Mask tensor, with True at the positions where targets were
            selected to be predicted
        """
        row_lengths = ids_seq.row_lengths(1)

        assertion_min_seq_length = tf.Assert(tf.reduce_all(row_lengths > 1), [row_lengths])

        with tf.control_dependencies([assertion_min_seq_length]):
            # Targets are masked according to a probability
            target_mask_by_prob = self._get_masked_by_prob(row_lengths, prob=self.masking_prob)
            # Exactly one target is masked per row
            one_target_mask = self._get_one_masked(row_lengths)

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
    def _get_masked_by_prob(row_lengths: tf.Tensor, prob: float) -> tf.Tensor:
        """Generates a dense mask boolean tensor with True values
        for randomly selected targets
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        batch_size = tf.shape(row_lengths)[0]
        output = tf.cast(random_bernoulli([batch_size, max_seq_length], p=prob), tf.bool)
        padding_mask = tf.sequence_mask(row_lengths)
        # Ignoring masked items in the padding positions
        output = tf.logical_and(output, padding_mask)
        return output

    @staticmethod
    def _get_one_masked(row_lengths: tf.Tensor):
        """Generates a dense mask boolean tensor where for each tensor (row)
        there is exactly one True value (selected target)
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        random_targets_indices = tf.cast(
            tf.math.floor(
                (
                    tf.random.uniform(
                        shape=tf.shape(row_lengths), minval=0.0, maxval=1.0, dtype=tf.float32
                    )
                    * tf.cast(row_lengths, tf.float32)
                )
            ),
            tf.int32,
        )

        one_target_mask = tf.cast(tf.one_hot(random_targets_indices, max_seq_length), tf.bool)
        return one_target_mask

    def get_config(self):
        config = super().get_config()
        config["masking_prob"] = self.masking_prob
        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        target = config.pop("target")
        masking_prob = config.pop("masking_prob")
        return cls(schema, target, masking_prob, **config)

    def configure_for_train(self):
        """Method called by the model.fit() to set the specialized
        `masking_post` and `masking_pre` needed by the TransformerBlock
        to align with the SequencePredictNext outputs.
        """
        if self.transformer is not None:
            if not is_transformers_available():
                raise ImportError("HuggingFace library `transformers` is required")
            from merlin.models.tf.transformers.transforms import (
                TransformerInferenceHiddenState,
                TransformerOutputToRagged,
            )

            # set the tansformer block with the correct masking blocks
            self.transformer.masking_post = combinators.SequentialBlock(
                [TransformerOutputToRagged(), TransformerInferenceHiddenState()]
            )
            self.transformer.masking_pre = combinators.SequentialBlock(
                [SequenceMaskLastInference(), ExtractMaskFromTargets(), ReplaceMaskedEmbeddings()]
            )

    def configure_for_test(self):
        """Method called by the model.evaluate() to check that the
        `masking_pre` set in the TransformerBlock is aligned with
        the evaluation strategy of SequenceMaskRandom
        """
        if self.transformer is not None:
            if self.transformer.masking_pre is None:
                raise ValueError(
                    "To evaluate using `SequenceMaskRandom`, ensure that your TransformerBlock has"
                    " `masking_pre` set as"
                    " `combinators.SequentialBlock("
                    "   ["
                    "        SequenceMaskLastInference(),"
                    "        ExtractMaskFromTargets(),"
                    "        ReplaceMaskedEmbeddings()"
                    "   ]"
                    ")`"
                    " You can automatically set `masking_pre` by passing `SequenceMaskRandom`"
                    " as the `pre` argument to the `fit` method:"
                    " `model.fit(..., pre=SequenceMaskRandom(...))`."
                )

            if not any(
                isinstance(layer, ReplaceMaskedEmbeddings)
                for layer in self.transformer.masking_pre.layers
            ):
                ValueError(
                    " The block `ReplaceMaskedEmbeddings` must be part of the `masking_pre`"
                    " of your TransformerBlock to be able to use `SequenceMaskRandom`"
                    " evaluation strategy."
                    " Please ensure that your Transformer model has been trained with"
                    " `SequenceMaskRandom` or `SequenceMaskLast`"
                    " by passing it as the `pre` argument to the `fit` method: "
                    "`model.fit(..., pre=SequenceMaskRandom(...))`."
                )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceMaskLast(SequenceTargetAsInput):
    """This block copies one of the sequence input features to be
    the target feature. The last item of the target (and corresponding
    sequences) is selected (masked) to be predicted
    The input and target masks are returned by using Keras Masking
    (._keras_mask), which is set by the compute_mask() method.

    Parameters
    ----------
    schema : Schema
        The input schema, that will be used to discover the name
        of the item id column
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    """

    def compute_mask(self, inputs, mask=None):
        """Selects (masks) the last position of the sequential targets
        to be predicted.
        This method is called by Keras after call()
        and returns the targets mask that will
        be assigned to the input tensors and targets, being accessible
        by tensor._keras_mask
        """
        item_id_seq = inputs[self.target_name]
        self.target_mask = self._generate_target_mask(item_id_seq)

        inputs_mask = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                inputs_mask[k] = self.target_mask
            else:
                inputs_mask[k] = None

        return (inputs_mask, self.target_mask)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor) -> tf.RaggedTensor:
        """Returns a bool ragged tensor with the last positions of the sequence masked

        Parameters
        ----------
        ids_seq : tf.RaggedTensor
            Sequence of ids, which are used to infer how many values
            each sequence contains

        Returns
        -------
        tf.RaggedTensor
            Mask tensor, with True at the last positions
        """
        row_lengths = ids_seq.row_lengths(1)
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)

        assertion_min_seq_length = tf.Assert(tf.reduce_all(row_lengths > 1), [row_lengths])

        with tf.control_dependencies([assertion_min_seq_length]):
            padding_mask = tf.sequence_mask(row_lengths)
            targets_mask = tf.ragged.boolean_mask(
                tf.cast(tf.one_hot(row_lengths - 1, max_seq_length), tf.bool), padding_mask
            )
            return targets_mask

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        target = config.pop("target")
        return cls(schema, target, **config)

    def configure_for_train(self):
        """Method called by the model.fit() to set the specialized
        `masking_post` and `masking_pre` needed by the TransformerBlock
        to align with the SequencePredictNext outputs.
        """
        if self.transformer is not None:
            if not is_transformers_available():
                raise ImportError("HuggingFace library `transformers` is required")
            from merlin.models.tf.transformers.transforms import (
                TransformerInferenceHiddenState,
                TransformerOutputToRagged,
            )

            # set the tansformer block with the correct masking blocks
            self.transformer.masking_post = combinators.SequentialBlock(
                [TransformerOutputToRagged(), TransformerInferenceHiddenState()]
            )
            self.transformer.masking_pre = combinators.SequentialBlock(
                [SequenceMaskLastInference(), ExtractMaskFromTargets(), ReplaceMaskedEmbeddings()]
            )

    def configure_for_test(self):
        """Method called by the model.evaluate() to check that the
        `masking_pre` set in the TransformerBlock is aligned with
        the evaluation strategy of SequenceMaskRandom
        """
        if self.transformer is not None:
            if self.transformer.masking_pre is None:
                raise ValueError(
                    "To evaluate using `SequenceMaskLast`, ensure that your TransformerBlock has"
                    " `masking_pre` set as"
                    " `combinators.SequentialBlock("
                    "   ["
                    "        SequenceMaskLastInference(),"
                    "        ExtractMaskFromTargets(),"
                    "        ReplaceMaskedEmbeddings()"
                    "   ]"
                    ")`"
                    " You can automatically set `masking_pre` by passing `SequenceMaskRandom`"
                    " or `SequenceMaskLast` as the `pre` argument to the `fit` method:"
                    " `model.fit(..., pre=SequenceMaskRandom(...))`."
                )

            if not any(
                isinstance(layer, ReplaceMaskedEmbeddings)
                for layer in self.transformer.masking_pre.layers
            ):
                ValueError(
                    "The block `ReplaceMaskedEmbeddings` must be part of the `masking_pre`"
                    " of your TransformerBlock to be able to use `SequenceMaskRandom`"
                    " evaluation strategy."
                    " Please ensure that your Transformer model has been trained with"
                    " `SequenceMaskRandom` or `SequenceMaskLast`"
                    " by passing it as the `pre` argument to the `fit` method: "
                    "`model.fit(..., pre=SequenceMaskLast(...))`."
                )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceMaskLastInference(Block):
    def call(self, inputs, training=False, testing=False):
        self.inference_mode = not training and not testing
        if self.inference_mode:
            # Extending sequences in one position by copying the last embedding
            repeat = inputs[:, -1:, :]
            # repeat = tf.expand_dims(repeat, 1)
            inputs = tf.concat([inputs, repeat], axis=1)
        return inputs

    def compute_mask(self, inputs, mask=None):
        """Selects (masks) the next position after the
        last valid (non-padded) position of the sequential targets
        to be predicted.
        This method is called by Keras after call()
        and returns the mask that is going to be assigned
        to the input tensors, being accessible
        by tensor._keras_mask
        """

        targets_mask = None
        if self.inference_mode:
            if isinstance(inputs, tf.RaggedTensor):
                row_lengths = inputs.row_lengths(1) + 1
                max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)

                padding_mask = tf.sequence_mask(row_lengths)
                targets_mask = tf.ragged.boolean_mask(
                    tf.cast(tf.one_hot(row_lengths - 1, max_seq_length), tf.bool), padding_mask
                )

        return targets_mask


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ReplaceMaskedEmbeddings(Block):
    """Takes a 3D input tensor (batch size x seq. length x embedding dim) and replaces
    by a dummy trainable single embedding at the positions to be masked.

    This is useful to be used when PredictMasked() transformation is used in
    the fit()/eval() methods, which randomly selects some targets to be predicted and uses
    Keras Masking to cascade the `_keras_mask`. By replacing input embeddings
    at masked positions we avoid target leakage when training models with
    Masked Language Modeling (BERT-like).

    To support masked training approach in Transformer-based model,
    SequenceMaskRandom and SequenceLastRandom implements `configure_for_train` method
    that sets `ReplaceMaskedEmbeddings` as part of the `masking_pre` of
    the transformer block.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        if self.hidden_size is None:
            raise ValueError("The last dim of inputs cannot be None")
        # Create a trainable embedding to replace masked interactions
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        self.masked_embedding = tf.Variable(initializer(shape=[self.hidden_size], dtype=tf.float32))

        return super().build(input_shape)

    def call(
        self,
        inputs: Union[tf.Tensor, tf.RaggedTensor],
    ) -> Union[tf.Tensor, tf.RaggedTensor]:
        """If the sequence of input embeddings is masked (with `tensor._keras_mask` defined),
        replaces the input embeddings for masked elements
        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.RaggedTensor]
            A tensor with sequences of vectors.
            Needs to be 3D (batch_size, sequence_length, embeddings dim).
            If inputs._keras_mask is defined uses it to infer the mask

        Returns
        -------
        Union[tf.Tensor, tf.RaggedTensor]
            returns a tensor with the masked inputs replaced by the dummy embedding
        """
        outputs = inputs
        if getattr(inputs, "_keras_mask", None) is not None:
            # Replaces the embeddings at masked positions by a dummy trainable embedding
            outputs = self._replace_masked_embeddings(inputs, inputs._keras_mask)
        return outputs

    def _replace_masked_embeddings(
        self, inputs: Union[tf.Tensor, tf.RaggedTensor], mask: Union[tf.Tensor, tf.RaggedTensor]
    ) -> tf.RaggedTensor:
        """
        Replaces in the inputs tensors the values masked as targets by a common trainable
        embedding
        """

        tf.Assert(
            tf_utils.check_inputs_mask_compatible_shape(inputs, mask),
            [
                "The inputs and mask need to be compatible: have the same dtype "
                "(tf.Tensor or tf.RaggedTensor) and the tf.rank(mask) == tf.rank(inputs)-1"
            ],
        )

        if isinstance(mask, tf.RaggedTensor):
            mask = mask.with_row_splits_dtype(inputs.row_splits.dtype)

        output = tf.where(
            tf.cast(tf.expand_dims(mask, -1), tf.bool),
            tf.cast(self.masked_embedding, dtype=inputs.dtype),
            inputs,
        )
        return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExtractMaskFromTargets(Block):
    """
    Recovers the mask information for the inputs from the mask information
    stored in the targets.

    This block looks for the Keras mask (`._keras_mask`) in the following order:
        1. Checks if the input tensor has a mask.
        2. Checks if there is a single target and if it has a mask.
        3. If there are multiple targets (dictionary), returns the mask of the target
        that matches the first two dimensions of the input.

    This is useful to use when the mask information for the inputs may be lost in
    previous non-mask-aware Merlin blocks.
    """

    def call(
        self,
        inputs: Union[tf.Tensor, tf.RaggedTensor],
        targets: Optional[Union[tf.Tensor, tf.RaggedTensor, TabularData]] = None,
    ) -> Union[tf.Tensor, tf.RaggedTensor]:
        mask = self._infer_mask_from_inputs_or_targets(inputs, targets)
        inputs._keras_mask = mask
        return inputs

    def _infer_mask_from_inputs_or_targets(
        self,
        inputs: Union[tf.Tensor, tf.RaggedTensor],
        targets: Optional[Union[tf.Tensor, tf.RaggedTensor]] = None,
    ):
        mask = None
        if getattr(inputs, "_keras_mask", None) is not None:
            mask = inputs._keras_mask

        elif targets is not None:
            if isinstance(targets, dict):
                if len(targets) == 1:
                    single_target = list(targets.values())[0]
                    if getattr(single_target, "_keras_mask", None) is not None:
                        mask = single_target._keras_mask
                elif len(targets) > 1:
                    # If there is more than 1 target, checks if only one
                    # target matches the shape sequence of the inputs
                    for _, v in targets.items():
                        if getattr(
                            v, "_keras_mask", None
                        ) is not None and tf_utils.check_inputs_mask_compatible_shape(
                            inputs, v._keras_mask
                        ):
                            if mask is None:
                                mask = v._keras_mask
                            else:
                                raise ValueError(
                                    "It is not possible to infer the mask "
                                    "from the targets because there are more than "
                                    "one target with the expected shape that matches "
                                    "the inputs shape (batch_size x seq_length)"
                                )
            elif getattr(targets, "_keras_mask", None) is not None:
                mask = targets._keras_mask

        return mask


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequenceCausalLastInference(Block):
    def call(self, inputs, training=False, testing=False):
        self.inference_mode = not training and not testing
        return inputs

    def compute_mask(self, inputs, mask=None):
        """Selects (masks) the last non padded position of the
        input sequence to be predicted.
        This method is called by Keras after call()
        and returns the mask that is going to be assigned
        to the input tensors, being accessible
        by tensor._keras_mask
        """
        if self.inference_mode:
            if isinstance(inputs, tf.RaggedTensor):
                row_lengths = inputs.row_lengths(1)
                max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)

                padding_mask = tf.sequence_mask(row_lengths)
                mask = tf.ragged.boolean_mask(
                    tf.cast(tf.one_hot(row_lengths - 1, max_seq_length), tf.bool), padding_mask
                )
        return mask
