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
import warnings
from itertools import combinations
from typing import Dict, Optional, Sequence, Union

import tensorflow as tf
from keras.layers.preprocessing import preprocessing_utils
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import layer_utils

from merlin.models.config.schema import requires_schema
from merlin.models.tf.core.base import Block, PredictionOutput
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import (
    df_to_tensor,
    get_candidate_probs,
    list_col_to_ragged,
    transform_label_to_onehot,
)
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags

ONE_HOT = utils.ONE_HOT
MULTI_HOT = utils.MULTI_HOT
COUNT = utils.COUNT


@Block.registry.register("as-ragged")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsRaggedFeatures(TabularBlock):
    """Convert all list (multi-hot/sequential) features to tf.RaggedTensor"""

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                outputs[name] = list_col_to_ragged(val)
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shapes):
        output_shapes = {}
        for k, v in input_shapes.items():
            # If it is a list/sparse feature (in tuple representation), uses the offset as shape
            if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                output_shapes[k] = tf.TensorShape([v[1][0], None])
            else:
                output_shapes[k] = v

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)


@Block.registry.register("as-sparse")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsSparseFeatures(TabularBlock):
    """Convert all list-inputs to sparse-tensors.

    By default, the dataloader will represent list-columns as a tuple of values & row-lengths.

    """

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                val = list_col_to_ragged(val)
            if isinstance(val, tf.RaggedTensor):
                outputs[name] = val.to_sparse()
            else:
                outputs[name] = val

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@Block.registry.register("as-dense")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsDenseFeatures(TabularBlock):
    """Convert all list-inputs to dense-tensors.

    By default, the dataloader will represent list-columns as a tuple of values & row-lengths.


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
                val = list_col_to_ragged(val)
            if isinstance(val, tf.RaggedTensor):
                if self.max_seq_length:
                    outputs[name] = val.to_tensor(shape=[None, self.max_seq_length])
                else:
                    outputs[name] = tf.squeeze(val.to_tensor())
            else:
                outputs[name] = tf.squeeze(val)

        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shape)
        outputs = {}

        for key, val in input_shape.items():
            if isinstance(val, tuple):
                outputs[key] = tf.TensorShape((batch_size, self.max_seq_length))
            else:
                outputs[key] = tf.TensorShape((batch_size))

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

        return outputs.copy_with_updates(predictions=predictions, targets=targets)


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


def reshape_categorical_input_tensor_for_encoding(
    input, feat_name, features_2d_last_dim, output_mode, ensure_1d_for_one_hot_mode=True
):
    output = input
    reshape_fn = tf.sparse.reshape if isinstance(output, tf.SparseTensor) else tf.reshape
    if ensure_1d_for_one_hot_mode and output_mode == ONE_HOT:
        if features_2d_last_dim.get(feat_name, None) == 1 or input.get_shape()[-1] == 1:
            output = reshape_fn(output, [-1])
        elif feat_name in features_2d_last_dim or (
            input.get_shape()[-1] is not None and len(input.get_shape()) == 2
        ):
            raise ValueError(
                "One-hot accepts input tensors that are squeezable to 1D, but received"
                f" a tensor with shape: {input.get_shape()}"
            )

    else:
        # if feat_name in features_2d_last_dim or len(input.get_shape()) == 2:
        if feat_name in features_2d_last_dim or (
            input.get_shape()[-1] is not None and len(input.get_shape()) == 2
        ):
            # Ensures that the shape is known to avoid error on graph mode
            new_shape = (-1, features_2d_last_dim.get(feat_name, input.get_shape()[-1]))
            output = reshape_fn(output, new_shape)
        else:
            expand_dims_fn = (
                tf.sparse.expand_dims if isinstance(output, tf.SparseTensor) else tf.expand_dims
            )
            output = expand_dims_fn(output, 1)

    return output


@Block.registry.register_with_multiple_names("category_encoding")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class CategoryEncoding(TabularBlock):
    """
    A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those inputs. Only
    categorical features with "CATEGORICAL" as Tag can be transformed, and other features without
    this Tag would be discarded.
    It outputs a TabularData (Dict of features), where each feature is a 2D tensor computed
    based on the outputmode.

    Parameters:
    ----------
    schema : Optional[Schema]
        The `Schema` with the input features
    output_mode: Optional[str]
        Specification for the output of the layer. Defaults to `"multi_hot"`. Values can be
        "one_hot", "multi_hot" or "count", configuring the transformation layer as follows:
        - "one_hot": Encodes each individual element in the input into a tensor with shape
            (batch_size, feature_cardinality), containing a 1 at the element index.
            It accepts both 1D tensor or 2D tensor if squeezable (i.e., if the last dimension is 1).
        - "multi_hot": Encodes each categorical value from the 2D input features into a
            multi-hot representation with shape (batch_size, feature_cardinality), with 1 at
            the indices present in the sequence and 0 for the other position.
            If 1D feature is provided, it behaves the same as "one_hot".
        - "count": also expects 2D tensor like `"multi_hot"` and outputs the features
            with shape (batch_size, feature_cardinality). But instead of returning "multi-hot"
            values, it outputs the frequency (count) of the number of items each item occurs
            in each sample.
    sparse: Optional[Boolean]
        If true, returns a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.
        Setting sparse=True is recommended for high-cardinality features, in order to avoid
        out-of-memory errors.
    count_weights: Optional[Union(tf.Tensor, tf.RaggedTensor, tf.SparseTensor)]
        count_weights is used to calculate weighted sum of times a token at that index appeared when
        `output_mode` is "count"
    """

    def __init__(
        self,
        schema: Schema = None,
        output_mode="one_hot",
        sparse=False,
        count_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema.select_by_tag(Tags.CATEGORICAL))
        self.sparse = sparse
        self.cardinalities = schema_utils.categorical_cardinalities(self.schema)
        # 'output_mode' must be one of (COUNT, ONE_HOT, MULTI_HOT)
        layer_utils.validate_string_arg(
            output_mode,
            allowable_strings=(COUNT, ONE_HOT, MULTI_HOT),
            layer_name="CategoryEncoding",
            arg_name="output_mode",
        )
        self.output_mode = output_mode
        self.sparse = sparse
        if count_weights is not None:
            if self.output_mode != COUNT:
                raise ValueError(
                    "`count_weights` is not used when `output_mode` is not "
                    "`'count'`. Received `count_weights={count_weights}`."
                )
            self.count_weights = utils.ensure_tensor(count_weights, self.compute_dtype)
        else:
            self.count_weights = None

        # Used to reshape 1D<->2Dtensors depending on the output_mode, when in graph mode
        self.features_2d_last_dim = {}

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        outputs = {}
        for name, depth in self.cardinalities.items():
            # Ensures the input is a Tensor, SparseTensor, then convert to Tensor
            if isinstance(inputs[name], tf.RaggedTensor):
                raise ValueError(
                    f"All `CategoryEncoding` inputs should not contain a RaggedTensor. Received "
                    f"{name} with type of {type(inputs[name])}"
                )

            assertion_min_rank = tf.Assert(
                tf.logical_and(
                    tf.greater_equal(tf.rank(inputs[name]), 1),
                    tf.less_equal(tf.rank(inputs[name]), 2),
                ),
                [
                    "`CategoryEncoding` only accepts 1D or 2D-shaped inputs, but got "
                    f"different rank for {name}"
                ],
            )

            outputs[name] = utils.ensure_tensor(inputs[name])

            if isinstance(outputs[name], tf.SparseTensor):
                max_value = tf.reduce_max(outputs[name].values)
                min_value = tf.reduce_min(outputs[name].values)
            else:
                max_value = tf.reduce_max(outputs[name])
                min_value = tf.reduce_min(outputs[name])
            condition = tf.logical_and(
                tf.greater(tf.cast(depth, max_value.dtype), max_value),
                tf.greater_equal(min_value, tf.cast(0, min_value.dtype)),
            )
            assertion_valid_values = tf.Assert(
                condition,
                [
                    "Input values must be in the range 0 <= values < num_tokens"
                    " with num_tokens={}".format(depth)
                ],
            )
            with tf.control_dependencies([assertion_min_rank, assertion_valid_values]):
                outputs[name] = reshape_categorical_input_tensor_for_encoding(
                    outputs[name],
                    name,
                    self.features_2d_last_dim,
                    self.output_mode,
                    ensure_1d_for_one_hot_mode=True,
                )

                outputs[name] = utils.encode_categorical_inputs(
                    outputs[name],
                    output_mode=self.output_mode,
                    depth=depth,
                    dtype=self.compute_dtype,
                    sparse=self.sparse,
                    count_weights=self.count_weights,
                )

                if not self.sparse and isinstance(outputs[name], tf.SparseTensor):
                    outputs[name] = tf.sparse.to_dense(outputs[name])
        return outputs

    def compute_output_shape(self, input_shapes):
        outputs = {}
        for key in self.schema.column_names:
            input_shape = input_shapes[key]
            if not input_shape:
                outputs[key] = tf.TensorShape([self.cardinalities[key]])
            else:
                outputs[key] = tf.TensorShape(input_shape[:1] + [self.cardinalities[key]])

                if len(input_shape) == 2:
                    self.features_2d_last_dim[key] = input_shape[-1]

        return outputs

    def get_config(self):
        config = super().get_config()
        if self.schema:
            config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        config.update(
            {
                "output_mode": self.output_mode,
                "sparse": self.sparse,
                "count_weights": self.count_weights.numpy() if self.count_weights else None,
            }
        )
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
        item_freq_probs: Union[tf.Tensor, Sequence] = None,
        is_prob_distribution: bool = False,
        reg_factor: float = 1.0,
        schema: Schema = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)

        self.reg_factor = reg_factor

        if item_freq_probs is not None:
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


@Block.registry.register("hashed_cross")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class HashedCross(TabularBlock):
    """A transformation block which crosses categorical features using the "hashing trick".
    Conceptually, the transformation can be thought of as: hash(concatenation of features) %
    num_bins
    Example usage::
    model_body = ParallelBlock(
                TabularBlock.from_schema(schema=cross_schema, pre=ml.HashedCross(cross_schema,
                                        num_bins = 1000)),
                is_input=True).connect(ml.MLPBlock([64, 32]))
    model = ml.Model(model_body, ml.BinaryClassificationTask("click"))
    Parameters
    ----------
        schema : Schema
            The `Schema` with the input features
        num_bins : int
            Number of hash bins.
        output_mode: string
            Specification for the output of the layer. Defaults to "one_hot".  Values can be "int",
            or "one_hot", configuring the layer as follows:
            - `"int"`: Return the integer bin indices directly.
            - `"one_hot"`: Encodes each individual element in the input into an array with the same
                size as `num_bins`, containing a 1 at the input's bin index.
        sparse: bool
            Boolean. Only applicable to `"one_hot"` mode. If True, returns a `SparseTensor` instead
            of a dense `Tensor`. Defaults to False.
        output_name: string
            Name of output feature, if not specified, default would be
            cross_<feature_name>_<feature_name>_<...>
        infer_num_bins: bool
            If True, num_bins would be set as the multiplier of feature cadinalities, if the
            multiplier is bigger than max_num_bins, then it would be cliped by max_num_bins
        max_num_bins: int
            Upper bound of num_bins, by default 100000.
    """

    def __init__(
        self,
        schema: Schema,
        num_bins: int = None,
        sparse: bool = False,
        output_mode: str = "one_hot",
        output_name: str = None,
        infer_num_bins: bool = False,
        max_num_bins: int = 100000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (not infer_num_bins) and (num_bins is None):
            raise ValueError(
                "num_bins is not given, and infer_num_bins is False, either of them "
                "is required, if you want to set fixed num_bins, then set infer_num_bins to False,"
                " and set num_bins to an integer value, if you want to infer num_bins from the "
                "mulplier of feature cardinalities, at the same time you can set the max_num_bins."
            )

        if not (output_mode in ["int", "one_hot", "multi_hot"]):
            raise ValueError("output_mode must be 'int', 'one_hot', or 'multi_hot'")
        self.schema = schema
        self.output_mode = output_mode
        self.sparse = sparse
        if not output_name:
            self.output_name = "cross"
            for name in sorted(schema.column_names):
                self.output_name = self.output_name + "_" + name
        else:
            self.output_name = output_name

        # Set num_bins
        if num_bins:
            self.num_bins = num_bins
        else:
            cardinalities = schema_utils.categorical_cardinalities(schema)
            inferred_num_bins_from_cardinalities_multiplier = 1.0
            for cardinality in cardinalities.values():
                inferred_num_bins_from_cardinalities_multiplier = (
                    inferred_num_bins_from_cardinalities_multiplier * cardinality
                )
            self.num_bins = int(min(max_num_bins, inferred_num_bins_from_cardinalities_multiplier))

        # Used to enforce the shape of 2D tensors depending on the output_mode, when in graph mode
        self.features_2d_last_dim = dict()

    def call(self, inputs):
        self._check_at_least_two_inputs()

        _inputs = {}
        for name in self.schema.column_names:

            assertion_min_rank = tf.Assert(
                tf.logical_and(
                    tf.greater_equal(tf.rank(inputs[name]), 1),
                    tf.less_equal(tf.rank(inputs[name]), 2),
                ),
                [
                    "`HashedCross` only accepts 1D or 2D-shaped inputs, but got "
                    f"different rank for {name}"
                ],
            )

            with tf.control_dependencies([assertion_min_rank]):
                _inputs[name] = reshape_categorical_input_tensor_for_encoding(
                    inputs[name],
                    name,
                    self.features_2d_last_dim,
                    self.output_mode,
                    ensure_1d_for_one_hot_mode=False,
                )

        # Perform the cross and convert to dense
        output = tf.sparse.cross_hashed(list(_inputs.values()), self.num_bins)

        if self.output_mode == ONE_HOT:
            output = tf.sparse.reshape(output, [-1])

        # Encode outputs.
        outputs = {}
        outputs[self.output_name] = preprocessing_utils.encode_categorical_inputs(
            output,
            output_mode=self.output_mode,
            depth=self.num_bins,
            sparse=self.sparse,
        )

        if not self.sparse and isinstance(outputs[self.output_name], tf.SparseTensor):
            outputs[self.output_name] = tf.sparse.to_dense(outputs[self.output_name])
        return outputs

    def compute_output_shape(self, input_shapes):
        self._check_at_least_two_inputs()
        self._check_input_shape_and_type(input_shapes)

        # Save the last dim for 2D features so that we can reshape them in graph mode in call()
        for key in self.schema.column_names:
            input_shape = input_shapes[key]
            if len(input_shape) == 2:
                self.features_2d_last_dim[key] = input_shape[-1]

        output_shape = {}
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        output_shape[self.output_name] = preprocessing_utils.compute_shape_for_encode_categorical(
            shape=[batch_size, 1], output_mode=self.output_mode, depth=self.num_bins
        )
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_bins": self.num_bins,
                "output_mode": self.output_mode,
                "sparse": self.sparse,
                "output_name": self.output_name,
            }
        )
        if self.schema:
            config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        return config

    def _check_at_least_two_inputs(self):
        if len(self.schema) < 2:
            raise ValueError(
                "`HashedCross` should be called on at least two features. "
                f"Received: {len(self.schema)} schemas"
            )
        for name, column_schema in self.schema.column_schemas.items():
            if Tags.CATEGORICAL not in column_schema.tags:
                warnings.warn(
                    f"Please make sure input features to be categorical, detect {name} "
                    "has no categorical tag"
                )

    def _check_input_shape_and_type(self, inputs_shapes) -> TabularData:
        _inputs_shapes = []
        for name in self.schema.column_names:
            shape = inputs_shapes[name]

            if shape.rank not in [1, 2]:
                raise ValueError(
                    "All `HashedCross` inputs should have 1D or 2D shape. "
                    f"Received: input {name} with shape={shape}"
                )

            _inputs_shapes.append(shape)

        if len(set([shape[0] for shape in _inputs_shapes])) > 1:
            raise ValueError(
                "All `HashedCross` inputs should have equal first dim (batch size). "
                f"Received: inputs={_inputs_shapes}"
            )


def HashedCrossAll(
    schema: Schema,
    num_bins: int = None,
    infer_num_bins: bool = False,
    max_num_bins: int = 100000,
    max_level: int = 2,
    sparse: bool = False,
    output_mode: str = "one_hot",
    ignore_combinations: Sequence[Sequence[str]] = [],
) -> Block:
    """Parallel block consists of HashedCross blocks for all combinations of schema with all levels
        through level 2 to max_level.

    Parameters:
    ----------
    schema: Schema
        Schema of the input data.
    max_level: int
        Max num of levels, this function would hash cross all combinations, the number of features
        included in these combinations is in the range from 2 to max_level, i.e. [2, max_level], by
        default 2, which means it would return hashed cross blocks of all level 2 combinations of
        features within schema

        For example, if schemas contain 3 features: feature_1, feature_2 and feature_3, if we call
            `level_3_cross = HashedCrossAll(schema = schemas, max_level = 3)`
        Then level_3_cross is a Parallel block, which contains 4 hashed crosses of
            1) feature_1 and feature_2
            2) feature_1 and feature_3
            3) feature_2 and feature_3
            4) feature_1, feature_2 and feature_3
    num_bins : int
        Number of hash bins, note that num_bins is for all hashed cross transformation block, no
        matter what level it is, if you want to set different num_bins for different hashed cross,
        please use HashedCross to define each one with different num_bins.
    output_mode: string
        Specification for the output of the layer. Defaults to
        `"one_hot"`.  Values can be `"int"`, or `"one_hot"` configuring the layer as
        follows:
        - `"int"`: Return the integer bin indices directly.
        - `"one_hot"`: Encodes each individual element in the input into an
            array the same size as `num_bins`, containing a 1 at the input's bin
            index.
    sparse : bool
        Boolean. Only applicable to `"one_hot"` mode. If True, returns a
        `SparseTensor` instead of a dense `Tensor`. Defaults to False.
    infer_num_bins: bool
        If True, all num_bins would be set as the multiplier of corresponding feature cadinalities,
        if the multiplier is bigger than max_num_bins, then it would be cliped by max_num_bins
    max_num_bins: int
        Upper bound of num_bins for all hashed cross transformation blocks, by default 100000.

    ignore_combinations :  Sequence[Sequence[str]]
        If provided, ignore feature combinations from this list.
        Useful to avoid interacting features whose combined value is always the same.
        For example, interacting these features is not useful and one of the features
        is dependent on the other :
        [["item_id", "item_category"], ["user_id", "user_birth_city", "user_age"]]


    Example usage::

        level_3_cross = HashedCrossAll(schema = schemas, max_level = 3, infer_num_bins = True)
    """

    if max_level < 2 or max_level > 3:
        raise ValueError(
            "Please make sure 1 < max_level < 4, because the cross transformation requires at "
            "least 2 features, and HashedCrossAll only support at most 3 level, if you want to "
            "cross more than 3 features, please use HashedCross. Received: max_level = {max_level}"
        )

    if len(schema) < 2:
        raise ValueError(
            "`HashedCrossing` should be called on at least two features. "
            f"Received: {len(schema)} schemas"
        )

    all_combinations = []
    for combination_tuple in combinations(schema.column_names, 2):
        all_combinations.append(set(combination_tuple))

    if max_level == 3:
        for combination_tuple in combinations(schema.column_names, 3):
            all_combinations.append(set(combination_tuple))

    if ignore_combinations:
        ignore_combinations_set = list([set(c) for c in ignore_combinations])
        all_combinations = list(
            [
                combination
                for combination in all_combinations
                if (combination not in ignore_combinations_set)
            ]
        )

    hashed_crosses = []
    for schema_names in all_combinations:
        hashed_crosses.append(
            HashedCross(
                schema=schema.select_by_name(schema_names),
                num_bins=num_bins,
                sparse=sparse,
                output_mode=output_mode,
                infer_num_bins=infer_num_bins,
                max_num_bins=max_num_bins,
            )
        )

    return ParallelBlock(hashed_crosses)
