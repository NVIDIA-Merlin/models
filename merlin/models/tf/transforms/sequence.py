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
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.backend import random_bernoulli

from merlin.models.tf.core.base import Block, BlockType, PredictionOutput
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.transforms.tensor import ListToRagged
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.constants import MASK_TARGETS_KEY
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
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        pre: Optional[BlockType] = None,
        **kwargs,
    ):
        _pre = ListToRagged()
        if pre:
            _pre = _pre.connect(pre)
        super().__init__(pre=_pre, schema=schema, **kwargs)

        self.target = target
        self.target_name = self._get_target(target)

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
        target_shape = inputs[self.target_name].get_shape().as_list()
        if len(target_shape) != 2:
            raise ValueError(
                f"The target column ({self.target_name}) is expected to be a 2D tensor,"
                f" but the shape is {target_shape}"
            )
        if target_shape[-1] == 1:
            raise ValueError(
                "The 2nd dim of the target column ({self.target_name}) should be greater"
                " than 1, so that the sequential input can be shifted as target"
            )

        seq_inputs_shapes = {
            col: inputs[col].get_shape().as_list() for col in self.schema.column_names
        }

        seq_shapes = list(seq_inputs_shapes.values())
        if not all(x == seq_shapes[0] for x in seq_shapes):
            raise ValueError(
                "The sequential inputs must have the same shape, but the shapes"
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
                    # Reducing 1 position of the seq length
                    new_input_shapes[k] = tf.TensorShape([v[0], v[1] - 1])

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
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
    """

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        # Shifts the target column to be the next item of corresponding input column
        new_target = inputs[self.target_name][:, 1:]
        if targets is None:
            targets = new_target
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            targets = dict()
            targets[self.target_name] = new_target

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                # Removes the last item of the sequence, as it belongs to the target
                new_inputs[k] = v[:, :-1]
            else:
                new_inputs[k] = v

        return (new_inputs, targets)


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
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be processed
    """

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Tuple:
        self._check_seq_inputs_targets(inputs)

        # Shifts the target column to be the next item of corresponding input column
        new_target = inputs[self.target_name][:, -1:]
        new_target = tf.squeeze(tf.sparse.to_dense(new_target.to_sparse()), axis=1)
        if targets is None:
            targets = new_target
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            targets = dict()
            targets[self.target_name] = new_target

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                # Removes the last item of the sequence, as it belongs to the target
                new_inputs[k] = v[:, :-1]
            else:
                new_inputs[k] = v

        return (new_inputs, targets)


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
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
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
        input_mask = positions_matrix < random_targets_indices
        target_mask = positions_matrix == random_targets_indices

        new_target = tf.squeeze(tf.ragged.boolean_mask(inputs[self.target_name], target_mask), 1)
        if targets is None:
            targets = new_target
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            targets = dict()
            targets[self.target_name] = new_target

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                new_inputs[k] = tf.ragged.boolean_mask(v, input_mask)
            else:
                new_inputs[k] = v

        return (new_inputs, targets)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequencePredictMasked(SequenceTransform):
    """This block implements the Masked Language Modeling (MLM) training approach
    introduced in BERT (NLP) and later adapted to RecSys by BERT4Rec [1].
    It is meant to be used as a transformation in the data loader
    (e.g. `Loader(..., transform=SequencePredictMasked())`).
    Given an input tf.RaggedTensor with sequences of embeddings
    and the corresponding sequence of item ids, some positions are randomly selected (masked)
    to be the targets for prediction.
    The targets are output being the same as the input ids sequence.
    The target masks can be returned in two different ways, depending on enable_keras_masking:
    (1) by using Keras Masking (._keras_mask) or (2) by
    including a special feature "__mask__", that contains a dict
    with the mask for the target. In the latter ExtractTargetsMask() block
    should be used in the model to extract the mask from special feature "__mask__"
    and make it available to following blocks/layers via Keras Masking (._keras_mask).
    This `enable_keras_masking=False` option is needed because the tensors _keras_mask set
    in the Loader transformation are lost when the model is trained.
    So the ExtractTargetsMask() transformation
    was created to convert the special input "__mask__" to Keras Masking (._keras_mask)
    inside the model, so that mask can be cascaded through the model layers/blocks

    Note: This transformation should be applied only during training, as you want
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
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    masking_prob : float, optional
        Probability of an item to be selected (masked) as a label of the given sequence.
        Note: We enforce that at least one item is masked for each sequence, so that it
        is useful for training, by default 0.2
    enable_keras_masking : bool, by default False
        If True, returns the masks in inputs and targets by using Keras Masking
        (._keras_mask), which is set by the compute_mask() method.
        If False, returns the target masks as a special input feature with key "__mask__".
        This option is needed because the _keras_mask set in the Loader transformation
        are lost when the model is trained. So the ExtractTargetsMask() transformation
        was created to convert the special input "__mask__" to Keras Masking (._keras_mask)
        inside the model, so that mask can be cascaded through the model layers/blocks
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        masking_prob: float = 0.2,
        enable_keras_masking: bool = False,
        **kwargs,
    ):
        self.masking_prob = masking_prob
        self.enable_keras_masking = enable_keras_masking
        super().__init__(schema, target, **kwargs)

    def call(
        self, inputs: TabularBlock, targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None
    ) -> Prediction:
        """Selects (masks) some positions from the input sequence features to be the targets
        and outputs as targets a copy of the items id sequence.
        It adds to the input features a dummy "__mask__" key, that contains a dict
        with the mask for the target.  The ExtractTargetsMask() should be used in the model to
        extract the mask from dummy feature "__mask__" and make it available
        to following blocks/layers via Keras Masking (._keras_mask)

        Parameters
        ----------
        inputs : TabularBlock
            A dict with the input features
        targets : Union[tf.Tensor, Dict[str, tf.Tensor]], optional
            The targets tensor or dict of tensors

        Returns
        -------
        Prediction
            Returns a Prediction(inputs, targets)
        """
        self._check_seq_inputs_targets(inputs)

        if self.target_name not in self.target_name:
            raise ValueError(
                f"The inputs provided does contain the target column ({self.target_name})"
            )

        new_target = tf.identity(inputs[self.target_name])
        if targets is None:
            targets = new_target
        else:
            if not isinstance(targets, dict):
                targets = dict()
            targets[self.target_name] = new_target

        target_mask = self._generate_target_mask(inputs[self.target_name])

        if not self.enable_keras_masking:
            self.save_mask_to_inputs(inputs, target_mask)

        return (inputs, targets)

    def save_mask_to_inputs(self, inputs, target_mask):
        if MASK_TARGETS_KEY not in inputs:
            inputs[MASK_TARGETS_KEY] = {self.target_name: target_mask}
        else:
            inputs[MASK_TARGETS_KEY][self.target_name] = target_mask

    def compute_mask(self, inputs, mask=None):
        """Is called by Keras and returns the targets mask that will
        be assigned to the input tensors and targets, being accessible
        by inputs._keras_mask
        """
        if not self.enable_keras_masking:
            return None

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
        config["enable_keras_masking"] = self.enable_keras_masking
        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        target = config.pop("target")
        masking_prob = config.pop("masking_prob")
        enable_keras_masking = config.pop("enable_keras_masking")
        return cls(schema, target, masking_prob, enable_keras_masking, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExtractTargetsMask(Block):
    """Extracts the target masks from the "__mask__" input

    Parameters
    ----------
    Block : _type_
        _description_
    """

    def call(self, inputs: TabularBlock, targets=None, features=None) -> Prediction:
        if (
            targets is not None
            and MASK_TARGETS_KEY in features
            and len(features[MASK_TARGETS_KEY]) > 0
        ):
            if isinstance(targets, dict):
                for k, v in targets.items():
                    v._keras_mask = features[MASK_TARGETS_KEY][k]
            else:
                if len(features[MASK_TARGETS_KEY]) == 1:
                    targets._keras_mask = list(features[MASK_TARGETS_KEY].values())[0]
                else:
                    raise ValueError(
                        "Many targets masks are provided "
                        f"({list(features[MASK_TARGETS_KEY].keys())})"
                        " as a dict target is not a dict."
                    )

        return Prediction(inputs, targets)

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class MaskSequenceEmbeddings(Block):
    """Takes a 3D input tensor (batch size x seq. length x embedding dim) and replaces
    by a dummy trainable single embedding at the positions to be masked.
    This block looks for the Keras mask (`._keras_mask`) in the following order:
      1. Checks if the input tensor has a mask
      2. Checks if there is a single target and if it has a mask
      3. If there are multiple targets (dict) returns the mask of the target
      that matches the first 2 dims of the input
    This is useful to be used when PredictMasked() transformation is used in
    the Loader, which randomly selects some targets to be predicted and uses
    Keras Masking to cascade the `_keras_mask`. By replacing input embeddings
    at masked positions we avoid target leakage when training models with
    Masked Language Modeling (BERT-like)
    """

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("The inputs must be a 3D tensor (batch_size, seq_length, vector_dim)")
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
        targets: Optional[Union[tf.Tensor, tf.RaggedTensor, TabularData]] = None,
        training: bool = False,
        features=None,
        testing: bool = False,
    ) -> Union[tf.Tensor, tf.RaggedTensor]:
        """Masks some items from the input sequence to be the targets
        and output the input tensor with replaced embeddings for masked
        elements and also the targets (copy of the items ids sequence)
        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.RaggedTensor]
            A tensor with sequences of vectors.
            Needs to be 3D (batch_size, sequence_length, embeddings dim).
            If inputs._keras_mask is defined uses it to infer the mask
        targets : Union[tf.Tensor, tf.RaggedTensor, TabularData], optional
            The target values, from which the mask can be extracted
            if targets inputs._keras_mask is defined.
        training : bool, optional
            A flag indicating whether model is being trained or not, by default False.
            If True, the masked positions of the inputs tensor are replaced by
            the dummy embedding
        Returns
        -------
        Union[tf.Tensor, tf.RaggedTensor]
            If training, returns a tensor with the masked inputs replaced by the dummy embedding
        """
        if len(inputs.shape.as_list()) != 3:
            raise ValueError("The inputs must be a 3D tensor (batch_size, seq_length, vector_dim)")

        outputs = inputs
        if training or testing:
            # Infers the mask from the inputs or targets
            mask = self._infer_mask_from_inputs_or_targets(inputs, targets)
            # Replaces the embeddings at masked positions by a dummy trainable embedding
            outputs = self._replace_masked_embeddings(inputs, mask)
        return outputs

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
                        ) is not None and self._check_inputs_mask_compatible_shape(
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

        if mask is None:
            raise ValueError("No valid mask was found on inputs or targets")

        if len(mask.shape.as_list()) != 2:
            raise ValueError(
                "The mask should be a 2D Tensor (batch_size x seq_length) but "
                f"its shape is {mask.shape}"
            )

        return mask

    def _check_inputs_mask_compatible_shape(
        self, inputs: Union[tf.Tensor, tf.RaggedTensor], mask: Union[tf.Tensor, tf.RaggedTensor]
    ):
        result = False
        if inputs.shape.as_list()[:2] == mask.shape.as_list()[:2]:
            if isinstance(inputs, tf.RaggedTensor):
                result = tf.reduce_all(
                    tf.cast(inputs.row_lengths(1), tf.int32)
                    == tf.cast(mask.row_lengths(1), tf.int32)
                )
            else:
                result = inputs.shape.as_list()[1] == mask.shape.as_list()[1]
        return result

    def _replace_masked_embeddings(
        self, inputs: Union[tf.Tensor, tf.RaggedTensor], mask: Union[tf.Tensor, tf.RaggedTensor]
    ) -> tf.RaggedTensor:
        """
        Replaces in the input tensor the values masked as targets by a common trainable
        embedding
        """

        if not (isinstance(inputs, tf.RaggedTensor) and isinstance(mask, tf.RaggedTensor)) and not (
            isinstance(inputs, tf.Tensor) and isinstance(mask, tf.Tensor)
        ):
            raise ValueError(
                "The inputs and mask need to be both either tf.Tensor or tf.RaggedTensor"
            )

        if isinstance(mask, tf.RaggedTensor):
            mask = mask.with_row_splits_dtype(inputs.row_splits.dtype)

        output = tf.where(
            tf.cast(tf.expand_dims(mask, -1), tf.bool),
            tf.cast(self.masked_embedding, dtype=inputs.dtype),
            inputs,
        )
        return output
