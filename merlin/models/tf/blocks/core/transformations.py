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
from typing import Dict, Optional, Sequence, Union

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops

from merlin.models.config.schema import requires_schema
from merlin.models.tf.blocks.core.base import Block, PredictionOutput
from merlin.models.tf.blocks.core.combinators import TabularBlock
from merlin.models.tf.typing import TabularData, TensorOrTabularData
from merlin.models.tf.utils.tf_utils import (
    df_to_tensor,
    get_candidate_probs,
    transform_label_to_onehot,
)
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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class LogitsTemperatureScaler(Block):
    """Scale the logits higher or lower,
    this is often used to reduce the overconfidence of the model.

    Parameters
    ----------
    temperature : float
        Divide the logits by this scaler.
    apply_on_call_outputs: bool
        Whether to apply the transform (logits / temperature) on
        `call()` or `call_outputs()`. By default True
    """

    def __init__(self, temperature: float, apply_on_call_outputs: bool = True, **kwargs):
        super(LogitsTemperatureScaler, self).__init__(**kwargs)
        self.temperature = temperature
        self.apply_on_call_outputs = apply_on_call_outputs

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        if not self.apply_on_call_outputs:
            return self.apply_temperature(inputs)
        else:
            return inputs

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        predictions = self.apply_temperature(predictions)

        return outputs.copy_with_updates(predictions=predictions, targets=targets)

    def compute_output_shape(self, input_shape):
        return input_shape

    def apply_temperature(self, predictions):
        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
        predictions = predictions / self.temperature
        return predictions


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
        self.item_domain = schema_utils.categorical_domains(schema)[self.item_id_feature_name]

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.num_classes,),
            initializer=self.bias_initializer,
        )
        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        embedding_table = self.context.get_embedding(self.item_domain)
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


@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class PopularityLogitsCorrection(Block):
    """Correct the predicted logit scores based on the item frequency,
    using the logQ correction proposed in sampled softmax [1]_ [2]_.
    The correction is done as `logits -= log(item_prob)`,
    where `item_prob = item_freq_count / sum(item_freq_count)` is
    a probability distribution of the item frequency. In a nutshell,
    the logQ correction aims to increase the prediction scores (logits)
    for infrequent items and decrease the ones for frequent items.

    References
    ----------
    .. [1] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic
       Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial
       Intelligence and Statistics (AISTATS).

    .. [2] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312

    Parameters:
    ----------
    item_freq_probs : Union[tf.Tensor, Sequence]
        A Tensor or list with item frequencies (if is_prob_distribution=False)
        or with item probabilities (if is_prob_distribution=True)
    is_prob_distribution: bool, optional
        If True, the item_freq_probs should be a probability distribution of the items.
        If False, the item frequencies is converted to probabilities
    reg_factor: float
        Factor to scale the logq correction, by default 1.0
    schema: Schema, optional
        The `Schema` with input features,
        by default None
    """

    def __init__(
        self,
        item_freq_probs: Union[tf.Tensor, Sequence],
        is_prob_distribution: bool = False,
        reg_factor: float = 1.0,
        schema: Schema = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)

        self.reg_factor = reg_factor

        self._check_items_cardinality(item_freq_probs)
        candidate_probs = get_candidate_probs(item_freq_probs, is_prob_distribution)

        self.candidate_probs = tf.Variable(
            candidate_probs,
            name="candidate_probs",
            trainable=False,
            dtype=tf.float32,
            validate_shape=False,
            shape=tf.shape(candidate_probs),
        )

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        frequencies_probs_col: str,
        is_prob_distribution: bool = False,
        gpu: bool = True,
        schema: Schema = None,
        **kwargs,
    ):
        """Load the item frequency table from a parquet file
        (in the format automatically generated by NVTabular with workflow.fit()).
        It supposed the parquet file has a single column with the item frequencies
        and is indexed by item ids.

        Parameters
        ----------
        parquet_path : str
            Path to the parquet file
        frequencies_probs_col : str
            Column name containing the items frequencies / probabilities
        is_prob_distribution: bool, optional
            If True, the frequencies_probs_col should contain the probability
            distribution of the items. If False, the frequencies_probs_col values
            are frequencies and will be converted to probabilities
        gpu : bool, optional
            Whether to load data using cudf, by default True
        schema: Schema, optional
            The `Schema` with input features,
            by default None

        Returns
        -------
            An instance of PopularityLogitsCorrection
        """
        # TODO: Use the schema to infer the path to the item frequency parquet table
        if gpu:
            import cudf

            df = cudf.read_parquet(parquet_path)
            item_frequency = tf.squeeze(df_to_tensor(df[frequencies_probs_col]))
        else:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            item_frequency = tf.squeeze(tf.convert_to_tensor(df[frequencies_probs_col].values))
        return cls(
            item_freq_probs=item_frequency,
            is_prob_distribution=is_prob_distribution,
            schema=schema,
            **kwargs,
        )

    def get_candidate_probs(self):
        return self.candidate_probs.value()

    def update(
        self, item_freq_probs: Union[tf.Tensor, Sequence], is_prob_distribution: bool = False
    ):
        """Updates the item frequencies / probabilities

        Parameters:
        ----------
        item_freq_probs : Union[tf.Tensor, Sequence]
            A Tensor or list with item frequencies (if is_prob_distribution=False)
            or with item probabilities (if is_prob_distribution=True)
        is_prob_distribution: bool, optional
            If True, the item_freq_probs should be a probability distribution of the items.
            If False, the item frequencies is converted to probabilities
        """
        self._check_items_cardinality(item_freq_probs)
        candidate_probs = get_candidate_probs(item_freq_probs, is_prob_distribution)
        self.candidate_probs.assign(candidate_probs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        predictions = outputs.predictions
        if training:
            positive_item_ids, negative_item_ids = (
                outputs.positive_item_ids,
                outputs.negative_item_ids,
            )
            positive_probs = tf.gather(self.candidate_probs, positive_item_ids)

            if negative_item_ids is not None:
                negative_probs = tf.gather(self.candidate_probs, negative_item_ids)
                # repeat negative scores for each positive item
                negative_probs = tf.reshape(
                    tf.tile(negative_probs, tf.shape(positive_item_ids)[0:1]),
                    (-1, tf.shape(negative_item_ids)[0]),
                )
                positive_probs = tf.concat(
                    [tf.expand_dims(positive_probs, -1), negative_probs], axis=1
                )

            # Applies the logQ correction
            epsilon = 1e-16
            predictions = predictions - (self.reg_factor * tf.math.log(positive_probs + epsilon))

        return outputs.copy_with_updates(predictions=predictions)

    def _check_items_cardinality(self, item_freq_probs):
        cardinalities = schema_utils.categorical_cardinalities(self.schema)
        item_id_feature_name = self.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        if tf.shape(item_freq_probs)[0] != cardinalities[item_id_feature_name]:
            raise ValueError(
                "The item frequency table length does not match the item ids cardinality"
                f"(expected {cardinalities[item_id_feature_name]}"
                f", got {tf.shape(item_freq_probs)[0]})"
            )
