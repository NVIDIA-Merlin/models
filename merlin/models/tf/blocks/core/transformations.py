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
from typing import Dict, Optional, Union

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops

from merlin.models.config.schema import requires_schema
from merlin.models.tf.blocks.core.base import Block, PredictionOutput
from merlin.models.tf.blocks.core.combinators import TabularBlock
from merlin.models.tf.typing import TabularData, TensorOrTabularData
from merlin.models.tf.utils.tf_utils import transform_label_to_onehot
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags


@Block.registry.register("as-sparse")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsSparseFeatures(TabularBlock):
    """
    Convert inputs to sparse tensors.
    """

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]

                if values.dtype.is_floating:
                    values = tf.cast(values, tf.int32)
                if row_lengths.dtype.is_floating:
                    row_lengths = tf.cast(row_lengths, tf.int32)

                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@Block.registry.register("as-dense")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsDenseFeatures(TabularBlock):
    """Convert sparse inputs to dense tensors

    Parameters
    ----------
    max_seq_length : int
        The maximum length of multi-hot features.
    """

    def __init__(self, max_seq_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                ragged = tf.RaggedTensor.from_row_lengths(values, row_lengths)
                if self.max_seq_length:
                    outputs[name] = ragged.to_tensor(shape=[None, self.max_seq_length])
                else:
                    outputs[name] = tf.squeeze(ragged.to_tensor())
            else:
                outputs[name] = tf.squeeze(val)

        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shape)
        outputs = {}

        for key, val in input_shape.items():
            if self.max_seq_length:
                outputs[key] = tf.TensorShape((batch_size, self.max_seq_length))
            else:
                # TODO: What to do here?
                raise ValueError("max_seq_length must be specified")

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_length": self.max_seq_length})

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RenameFeatures(TabularBlock):
    """Rename input features

    Parameters
    ----------
    renames: dict
        Mapping with new features names.
    schema: Schema, optional
        The `Schema` with input features,
        by default None
    """

    def __init__(
        self, renames: Dict[Union[str, Tags], str], schema: Optional[Schema] = None, **kwargs
    ):
        super().__init__(schema=schema, **kwargs)
        self.renames = {}
        for key, val in renames.items():
            if isinstance(key, Tags):
                if schema is None:
                    raise ValueError("Schema must be provided to rename features with Tags")
                cols = schema.select_by_tag(key)
                if len(cols) != 1:
                    raise ValueError(f"Tag: {key} does not uniquely identify a column")
                self.renames[cols.first.name] = val
            else:
                self.renames[key] = val

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}

        for key, val in inputs.items():
            if key in self.renames:
                outputs[self.renames[key]] = val
            else:
                outputs[key] = val

        return outputs

    def compute_output_shape(self, input_shape):
        outputs = {}

        for key, val in input_shape.items():
            if key in self.renames:
                outputs[self.renames[key]] = val
            else:
                outputs[key] = val

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"renames": self.renames})

        return config


@Block.registry.register_with_multiple_names("stochastic-swap-noise", "ssn")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class StochasticSwapNoise(TabularBlock):
    """
    Applies Stochastic replacement of sequence features
    """

    def __init__(self, schema=None, pad_token=0, replacement_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema
        self.pad_token = pad_token
        self.replacement_prob = replacement_prob

    def call(
        self,
        inputs: TensorOrTabularData,
        input_mask: Optional[tf.Tensor] = None,
        training=False,
        **kwargs,
    ) -> TensorOrTabularData:
        def augment(input_mask):
            if self._schema:
                input_mask = input_mask or self.get_padding_mask_from_item_id(
                    inputs, self.pad_token
                )

            if isinstance(inputs, dict):
                return {key: self.augment(val, input_mask) for key, val in inputs.items()}

            return self.augment(inputs, input_mask)

        output = control_flow_util.smart_cond(training, lambda: augment(input_mask), lambda: inputs)

        return output

    def augment(self, input_tensor: tf.Tensor, mask: Optional[tf.Tensor], **kwargs) -> tf.Tensor:
        if mask is not None:
            if len(input_tensor.shape) == len(mask.shape) - 1:
                mask = mask[:, 0]

        casted = tf.cast(
            backend.random_binomial(array_ops.shape(input_tensor), p=self.replacement_prob),
            tf.int32,
        )

        replacement_mask_matrix = casted * tf.cast(mask, tf.int32)

        n_values_to_replace = tf.reduce_sum(replacement_mask_matrix)

        input_flattened_non_zero = tf.boolean_mask(
            input_tensor, tf.cast(replacement_mask_matrix, tf.bool)
        )

        sampled_values_to_replace = tf.gather(
            input_flattened_non_zero,
            tf.random.shuffle(tf.range(tf.shape(input_flattened_non_zero)[0]))[
                :n_values_to_replace
            ],
        )

        replacement_indices = tf.sparse.from_dense(replacement_mask_matrix).indices

        output_tensor = tf.tensor_scatter_nd_update(
            input_tensor, replacement_indices, sampled_values_to_replace
        )

        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config["pad_token"] = self.pad_token
        config["replacement_prob"] = self.replacement_prob

        return config


@Block.registry.register_with_multiple_names("continuous-powers")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ContinuousPowers(TabularBlock):
    """Trick from `Deep Neural Networks for YouTube Recommendations`"""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs: TabularData = {}

        for key, val in inputs.items():
            outputs[key] = val
            if len(val.shape) < 2 or (len(val.shape) == 2 and val.shape[1] == 1):
                val_float = tf.cast(val, tf.float32)
                outputs[f"{key}_sqrt"] = tf.sqrt(val_float)
                outputs[f"{key}_pow"] = tf.pow(val_float, 2)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = {}

        for key, val in input_shape.items():
            output_shape[key] = val
            if len(val) < 2 or (len(val) == 2 and val[1] == 1):
                output_shape[f"{key}_sqrt"] = val
                output_shape[f"{key}_squared"] = val

        return output_shape


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExpandDims(TabularBlock):
    """
    Expand dims of selected input tensors.
    Example::

        inputs = {
            "cont_feat1": tf.random.uniform((NUM_ROWS,)),
            "cont_feat2": tf.random.uniform((NUM_ROWS,)),
            "multi_hot_categ_feat": tf.random.uniform(
                (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
            ),
        }

        expand_dims_op = tr.ExpandDims(expand_dims={"cont_feat2": 0, "multi_hot_categ_feat": 1})
        expanded_inputs = expand_dims_op(inputs)
    """

    def __init__(self, expand_dims: Union[int, Dict[str, int]] = -1, **kwargs):
        """Instantiates the `ExpandDims` transformation, which allows to expand dims
        of the input tensors

        Parameters
        ----------
        expand_dims : Union[int, Dict[str, int]], optional, by default -1
            Defines which dimensions should be expanded. If an `int` is provided, all input tensors
            will have the same dimension expanded. If a `dict` is passed, only features matching
            the dict keys will be expanded, in the dimension specified as the dict values.
        """
        super().__init__(**kwargs)
        self.inputs_expand_dims = expand_dims

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}

        for k, v in inputs.items():
            if isinstance(self.inputs_expand_dims, int):
                outputs[k] = tf.expand_dims(v, self.inputs_expand_dims)
            elif isinstance(self.inputs_expand_dims, dict) and k in self.inputs_expand_dims:
                expand_dim = self.inputs_expand_dims[k]
                outputs[k] = tf.expand_dims(v, expand_dim)
            elif self.inputs_expand_dims:
                outputs[k] = v
            else:
                raise ValueError("The expand_dims argument is not valid")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@Block.registry.register_with_multiple_names("l2-norm")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class L2Norm(TabularBlock):
    """Apply L2-normalization to input tensors along a given axis"""

    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)

    def call(self, inputs: Union[tf.Tensor, TabularData], axis: int = -1, **kwargs):
        if isinstance(inputs, dict):
            inputs = {key: tf.linalg.l2_normalize(inp, axis=axis) for key, inp in inputs.items()}
        else:
            inputs = tf.linalg.l2_normalize(inputs, axis=axis)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


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
        return outputs.copy_with_updates(
            predictions=predictions,
            targets=targets,
        )


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
class LogitsTemperatureScaler(Block):
    """Scale the logits higher or lower,
    this is often used to reduce the overconfidence of the model.

    Parameters
    ----------
    temperature : float
        Divide the logits by this scaler.
    """

    def __init__(self, temperature: float, **kwargs):
        super(LogitsTemperatureScaler, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs, training=True, **kwargs) -> tf.Tensor:
        if not training:
            assert isinstance(inputs, tf.Tensor), "Predictions must be a tensor"
            return inputs / self.temperature
        else:
            return inputs

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        if training:
            assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
            predictions = predictions / self.temperature

        return outputs.copy_with_updates(predictions=predictions, targets=targets)

    def compute_output_shape(self, input_shape):
        return input_shape


@Block.registry.register_with_multiple_names("weight-tying")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemsPredictionWeightTying(Block):
    """Tying the item embedding weights with the output projection layer matrix [1]
    The output logits are obtained by multiplying the output vector by the item-ids embeddings.

    Parameters
    ----------
        schema : Schema
            The `Schema` with the input features
        bias_initializer : str, optional
            Initializer to use on the bias vector, by default "zeros"

    References:
    -----------
    [1] Hakan, Inan et al.
        "Tying word vectors and word classifiers: A loss framework for language modeling"
        arXiv:1611.01462
    """

    def __init__(self, schema: Schema, bias_initializer="zeros", **kwargs):
        super(ItemsPredictionWeightTying, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        self.item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.num_classes = schema_utils.categorical_cardinalities(schema)[self.item_id_feature_name]

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


@Block.registry.register_with_multiple_names("categorical_to_onehot")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class CategoricalOneHot(Block):
    """
    Transform categorical features (2-D and 3-D tensors) to a one-hot representation.

    Parameters:
    ----------
    schema : Optional[Schema]
        The `Schema` with the input features
    """

    def __init__(self, schema: Schema = None, **kwargs):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)
        self.cardinalities = schema_utils.categorical_cardinalities(self.schema)

    def build(self, input_shapes):
        super(CategoricalOneHot, self).build(input_shapes)

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in self.cardinalities.items():
            outputs[name] = tf.squeeze(tf.one_hot(inputs[name], val))
        return outputs

    def compute_output_shape(self, input_shape):
        outputs = {}
        for key, val in input_shape.items():
            if len(val) == 3:
                outputs[key] = tf.TensorShape((val[0], val[1], self.cardinalities[key]))
            else:
                outputs[key] = tf.TensorShape((val[0], self.cardinalities[key]))
        return outputs

    def get_config(self):
        config = super().get_config()
        if self.schema:
            config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        return config


@Block.registry.register_with_multiple_names("label_to_onehot")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class LabelToOneHot(Block):
    """Transform the categorical encoded labels into a one-hot representation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions

        num_classes = tf.shape(predictions)[-1]
        targets = transform_label_to_onehot(targets, num_classes)

        return outputs.copy_with_updates(targets=targets)
