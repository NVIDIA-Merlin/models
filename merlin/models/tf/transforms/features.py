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
from typing import Dict, Optional, Sequence, Tuple, Union

import tensorflow as tf
from keras.layers.preprocessing import preprocessing_utils as p_utils
from keras.utils import layer_utils

from merlin.models.config.schema import requires_schema
from merlin.models.tf.core.base import Block, PredictionOutput
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.transforms.tensor import to_dense, to_sparse
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.tf.utils.tf_utils import list_col_to_ragged
from merlin.models.utils import schema_utils
from merlin.schema import ColumnSchema, Schema, Tags

ONE_HOT = p_utils.ONE_HOT
MULTI_HOT = p_utils.MULTI_HOT
COUNT = p_utils.COUNT


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class FeaturesTensorTypeConversion(TabularBlock):
    """Base class to convert the tensor type of features provided in the schema

    Parameters
        ----------
        schema : Optional[Schema], optional
            The schema with the columns that will be transformed, by default None
    """

    def __init__(self, schema: Optional[Schema] = None, **kwargs):
        super().__init__(schema=schema, **kwargs)
        self.column_names = None

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        raise NotImplementedError("The call method need to be implemented by child classes")

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, dict):
            return input_shape

        outputs = {}
        for name, val in input_shape.items():
            if self.has_schema and name in self.schema.column_names:
                col_schema_shape = self.schema[name].shape
                if col_schema_shape.is_list:
                    max_seq_length = col_schema_shape.dims[1].max
                    if max_seq_length is not None:
                        max_seq_length = int(max_seq_length)
                    shape = val.as_list()
                    if max_seq_length is not None:
                        max_seq_length = int(max_seq_length)
                    shape[1] = max_seq_length
                    val = tf.TensorShape(shape)

            outputs[name] = val
        return outputs

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        return input_schema


@Block.registry.register("to_sparse")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ToSparse(FeaturesTensorTypeConversion):
    """Convert the features provided in the schema to sparse tensors.
    The other features are kept unchanged.
    """

    def call(self, inputs: Union[TabularData, tf.RaggedTensor, tf.Tensor], **kwargs) -> TabularData:
        if isinstance(inputs, (tf.RaggedTensor, tf.SparseTensor)):
            return to_sparse(inputs)

        outputs = {}
        for name, val in inputs.items():
            if not self.has_schema or name in self.schema.column_names:
                val = to_sparse(val)

            outputs[name] = val

        return outputs


@Block.registry.register("to_dense")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ToDense(FeaturesTensorTypeConversion):
    """Convert the features provided in the schema to dense tensors.
    The other features are kept unchanged.
    """

    def call(
        self, inputs: Union[TabularData, tf.RaggedTensor, tf.SparseTensor], **kwargs
    ) -> TabularData:
        if isinstance(inputs, (tf.RaggedTensor, tf.SparseTensor)):
            return to_dense(inputs)

        outputs = {}
        for name, val in inputs.items():
            if self.has_schema:
                if name in self.schema.column_names:
                    col_schema_shape = self.schema[name].shape
                    if col_schema_shape.is_list:
                        # TODO: Not sure if we really need to extract the
                        # max seq length from the col schema to
                        # have a dense tensor properly set
                        max_seq_length = col_schema_shape.dims[1].max
                        if max_seq_length is not None:
                            max_seq_length = int(max_seq_length)
                        val = to_dense(val, max_seq_length=max_seq_length)
                    else:
                        val = to_dense(val)
            else:
                val = to_dense(val)

            outputs[name] = val

        return outputs


@Block.registry.register("prepare_lists")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PrepareListFeatures(TabularBlock):
    """Prepares all list (multi-hot/sequential) features,
    so that they converted to tf.RaggedTensor or dense tf.Tensor
    based on the columns schema.
    It manages in particular the Merlin dataloader representation of
    ragged list features, which consists of two keys in the inputs dict
    suffixed by "__values" and "__offsets".
    For example, the "categories" ragged list feature is represented
    by the Merlin Dataloader as "categories__values" and
    "categories__offsets" keys and this block converts them
    to a single tf.RaggedTensor "categories".

    Parameters
    ----------
    schema : Schema
        The features schema
    list_as_dense : bool, optional
        Whether to enforce all list features to be dense tf.Tensor (including
        the ragged ones), by default False
    """

    def __init__(self, schema: Schema, list_as_dense: Optional[bool] = False, **kwargs):
        super().__init__(schema=schema, **kwargs)
        self.list_as_dense = list_as_dense

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        if not inputs:
            return inputs

        outputs = {}
        if self.has_schema:
            for name in self.schema.column_names:
                val = None
                col_schema_shape = self.schema[name].shape
                if col_schema_shape.is_list:
                    if name in inputs:
                        val = inputs[name]
                        if isinstance(val, tf.RaggedTensor):
                            val = inputs[name]
                        elif isinstance(val, tf.SparseTensor):
                            val = tf.RaggedTensor.from_sparse(val)
                    else:
                        if col_schema_shape.is_ragged:
                            if f"{name}__values" not in inputs or f"{name}__offsets" not in inputs:
                                raise ValueError(
                                    f"The ragged list feature '{name}' is expected to be "
                                    f"represented by two features in the inputs: '{name}__values' "
                                    f"and '{name}__offsets', but they were not found."
                                )
                            val = list_col_to_ragged(
                                inputs[f"{name}__values"], inputs[f"{name}__offsets"]
                            )
                            del inputs[f"{name}__values"]
                            del inputs[f"{name}__offsets"]
                        else:
                            raise ValueError(f"Feature '{name}' was not found in the inputs")

                    if len(val.shape) == 2:
                        val = tf.expand_dims(val, axis=-1)

                    if self.list_as_dense:
                        val = to_dense(val, max_seq_length=col_schema_shape.dims[1].max)

                elif name in inputs:
                    val = inputs[name]

                if val is not None:
                    outputs[name] = val

        # Adding other inputs that might not be in the schema,
        # as they might be treated by other block
        for k, v in inputs.items():
            if k not in outputs:
                outputs[k] = v

        return outputs

    def compute_output_shape(self, input_shapes):
        if not input_shapes:
            return input_shapes

        output_shapes = {}
        if self.has_schema:
            for name in self.schema.column_names:
                col_schema_shape = self.schema[name].shape
                if col_schema_shape.is_list:
                    if name not in input_shapes and f"{name}__offsets" in input_shapes:
                        batch_size = input_shapes[f"{name}__offsets"][0]
                        if batch_size is not None:
                            # The length of offset is always batch size + 1
                            batch_size -= 1

                        seq_length = None
                        if self.list_as_dense or not self.schema[name].shape.is_ragged:
                            if col_schema_shape.dims[1] is None:
                                raise Exception(
                                    f"List feature {name} does not have maximum"
                                    "length set in the schema ({col_schema_shape})"
                                )
                            seq_length = int(col_schema_shape.dims[1].max)

                        output_shapes[name] = tf.TensorShape([batch_size, seq_length, 1])

                    else:
                        if len(input_shapes[name]) == 2:
                            output_shapes[name] = input_shapes[name] + (1,)

                elif name in input_shapes:
                    output_shapes[name] = input_shapes[name]

        # Adding other inputs that might not be in the schema,
        # as they might be treated by other block
        for k, v in input_shapes.items():
            if k.replace("__values", "").replace("__offsets", "") not in output_shapes:
                output_shapes[k] = v

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    def get_config(self):
        config = super().get_config()
        config.update({"list_as_dense": self.list_as_dense})

        return config


@Block.registry.register("prepare_features")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PrepareFeatures(TabularBlock):
    """Prepares scalar and list (multi-hot/sequential) features
    to be used with a Merlin model. The transformations are applied
    only for features in the schema, the other features are kept the same.
    The scalar features are extended to be 2D (batch_size, 1) and
    list features are converted to either tf.RaggedTensor or dense tf.Tensor
    based on the columns schema.
    It manages in particular the Merlin dataloader representation of list
    features, which consists of two keys in the inputs dict
    suffixed by "__values" and "__offsets".
    For example, the "categories" list feature are represented
    by the Merlin Dataloader as "categories__values" and
    "categories__offsets" keys and this block converts them
    to a single ragged or dense tensor "categories".

    Parameters
    ----------
    schema : Schema
        The features schema
    list_as_dense : bool, optional
        Whether to enforce all list features to be dense tf.Tensor (including
        the ragged ones), by default False
    """

    def __init__(self, schema: Schema, list_as_dense: Optional[bool] = False, **kwargs):
        super().__init__(schema=schema, **kwargs)
        self.list_as_dense = list_as_dense
        self.prepare_lists = PrepareListFeatures(schema, list_as_dense)

    def call(
        self, inputs: TabularData, targets: TabularData = None, **kwargs
    ) -> Union[TabularData, Tuple[TabularData, TabularData]]:
        # This might be needed for list targets prepared in the preprocessing
        # targets = self.prepare_lists(targets)
        inputs = self.prepare_lists(inputs)
        if not inputs:
            return inputs

        outputs = {}
        out_targets = {}
        if self.has_schema:
            # Preparing non-list features and targets
            for name in self.schema.column_names:
                if name in inputs:
                    val = inputs[name]

                    if not self.schema[name].shape.is_list:
                        # Expanding / setting last dim of non-list input features to be 2D
                        val = tf.reshape(val, (-1, 1))

                    outputs[name] = val

                if isinstance(targets, dict) and name in targets:
                    val = targets[name]

                    if not self.schema[name].shape.is_list:
                        # Expanding / setting last dim of non-list target features to be 2D
                        val = tf.reshape(val, (-1, 1))

                    out_targets[name] = val

        # Adding other inputs that might not be in the schema,
        # as they might be treated by other block
        for k, v in inputs.items():
            if k not in outputs:
                outputs[k] = v

        if targets is not None:
            if isinstance(targets, dict):
                for k, v in targets.items():
                    if k not in out_targets:
                        out_targets[k] = v
                        if v.get_shape().rank == 1:
                            # Expanding / setting last dim of non-list
                            # target features to be 2D
                            out_targets[k] = tf.reshape(v, (-1, 1))
            elif targets.get_shape().rank == 1:
                # Expanding / setting last dim of non-list
                # target features to be 2D
                out_targets = tf.reshape(targets, (-1, 1))
            else:
                out_targets = targets
            return (outputs, out_targets)
        return outputs

    def compute_output_shape(self, input_shapes):
        output_shapes = self.prepare_lists.compute_output_shape(input_shapes)

        if self.has_schema:
            for name in self.schema.column_names:
                if not self.schema[name].shape.is_list and name in input_shapes:
                    output_shapes[name] = tf.TensorShape([input_shapes[name][0], 1])

        # Adding other inputs that might not be in the schema,
        # as they might be treated by other block
        for k, v in input_shapes.items():
            if (
                k not in output_shapes
                and not k.endswith("__values")
                and not k.endswith("__offsets")
            ):
                output_shapes[k] = v

        return output_shapes

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    def get_config(self):
        config = super().get_config()
        config.update({"list_as_dense": self.list_as_dense})

        return config


@Block.registry.register("as-ragged")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Rename(TabularBlock):
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
            self.count_weights = p_utils.ensure_tensor(count_weights, self.compute_dtype)
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
                    tf.greater_equal(tf.rank(inputs[name]), 2),
                    tf.less_equal(tf.rank(inputs[name]), 3),
                ),
                [
                    "`CategoryEncoding` only accepts 2D (batch_size,1)"
                    " or 3D (batch_size,seq_length,1) inputs, but got different "
                    "rank for {name}"
                ],
            )

            reshape_fn = (
                tf.sparse.reshape if isinstance(inputs[name], tf.SparseTensor) else tf.reshape
            )
            inputs[name] = reshape_fn(inputs[name], tf.shape(inputs[name])[:-1])

            outputs[name] = p_utils.ensure_tensor(inputs[name])

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

                outputs[name] = p_utils.encode_categorical_inputs(
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
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        outputs = {}
        for key in self.schema.column_names:
            if key in input_shapes:
                outputs[key] = tf.TensorShape([batch_size, self.cardinalities[key]])

                input_shape = input_shapes[key]
                if input_shape[-1] == 1:
                    input_shape = input_shape[:-1]
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


@Block.registry.register_with_multiple_names("label_to_onehot")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ToOneHot(Block):
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
        targets = tf_utils.transform_label_to_onehot(targets, num_classes)

        return outputs.copy_with_updates(targets=targets)


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
            reshape_fn = (
                tf.sparse.reshape if isinstance(inputs[name], tf.SparseTensor) else tf.reshape
            )
            inputs[name] = reshape_fn(inputs[name], tf.shape(inputs[name])[:-1])

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
        outputs[self.output_name] = p_utils.encode_categorical_inputs(
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
            if input_shape[-1] == 1:
                input_shape = input_shape[:-1]
            if len(input_shape) == 2:
                self.features_2d_last_dim[key] = input_shape[-1]

        output_shape = {}
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        output_shape[self.output_name] = p_utils.compute_shape_for_encode_categorical(
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

            if shape[-1] == 1:
                shape = shape[:-1]

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
) -> ParallelBlock:
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


@Block.registry.register_with_multiple_names("to_target")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ToTarget(Block):
    """Transform columns to targets"""

    def __init__(
        self,
        schema: Schema,
        *target: Union[ColumnSchema, Schema, str, Tags],
        one_hot: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.schema = schema
        self.target = target
        self.one_hot = one_hot

    def _target_column_schemas(self):
        target_columns = {}
        for t in self.target:
            if isinstance(t, str):
                target_columns[t] = self.schema.select_by_name(t).first
            elif isinstance(t, ColumnSchema):
                target_columns[t.name] = t
            elif isinstance(t, Tags):
                selected_schema = self.schema.select_by_tag(t)
                for col_name, col_schema in selected_schema.column_schemas.items():
                    target_columns[col_name] = col_schema
            elif isinstance(t, Schema):
                for col_name, col_schema in t.column_schemas.items():
                    target_columns[col_name] = col_schema
            else:
                raise ValueError(f"Unsupported target type {type(t)}")
        return target_columns

    def call(
        self, inputs: TabularData, targets=None, **kwargs
    ) -> Tuple[TabularData, Union[TabularData, tf.Tensor, None]]:
        target_columns = self._target_column_schemas()

        outputs = {}
        for name in inputs:
            if name not in target_columns:
                outputs[name] = inputs[name]
                continue
            if isinstance(targets, dict):
                _target = targets.get(name, inputs[name])
                if self.one_hot:
                    _target = self._to_one_hot(name, _target)
                targets[name] = _target
            else:
                _target = inputs[name]
                if self.one_hot:
                    _target = self._to_one_hot(name, _target)
                targets = _target

        return outputs, targets

    def _to_one_hot(self, name, target):
        num_classes = schema_utils.categorical_cardinalities(self.schema)[name]
        one_hot = tf.one_hot(target, num_classes, dtype=target.dtype)
        return tf.squeeze(one_hot)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        target_columns = self._target_column_schemas()
        output_column_schemas = {}
        for col_name, col_schema in input_schema.column_schemas.items():
            if col_name in target_columns:
                output_column_schemas[col_name] = col_schema.with_tags(Tags.TARGET)
            else:
                output_column_schemas[col_name] = col_schema
        return Schema(column_schemas=output_column_schemas)


def reshape_categorical_input_tensor_for_encoding(
    input, feat_name, features_2d_last_dim, output_mode, ensure_1d_for_one_hot_mode=True
):
    output = input
    reshape_fn = tf.sparse.reshape if isinstance(output, tf.SparseTensor) else tf.reshape
    if ensure_1d_for_one_hot_mode and output_mode == ONE_HOT:
        if features_2d_last_dim.get(feat_name, None) == 1 or input.get_shape()[-1] == 1:
            output = reshape_fn(output, [-1])
        elif feat_name in features_2d_last_dim or (
            input.get_shape()[-1] is not None and input.get_shape().rank == 2
        ):
            raise ValueError(
                "One-hot accepts input tensors that are squeezable to 1D, but received"
                f" a tensor with shape: {input.get_shape()}"
            )

    else:
        if feat_name in features_2d_last_dim or (
            input.get_shape()[-1] is not None and len(input.get_shape()) == 2
        ):
            # Ensures that the shape is known to avoid error on graph mode
            new_shape = (-1, features_2d_last_dim.get(feat_name, None) or input.get_shape()[-1])
            output = reshape_fn(output, new_shape)
        else:
            expand_dims_fn = (
                tf.sparse.expand_dims if isinstance(output, tf.SparseTensor) else tf.expand_dims
            )
            output = expand_dims_fn(output, 1)

    return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BroadcastToSequence(Block):
    """Broadcast context features to match the timesteps of sequence features.

    This layer supports mask propagation. If the sequence features have a mask. The
    context features being broadcast will inherit the mask.

    Parameters
    ----------
    context_schema : Schema
        The schema representing contextual features to be broadcast
    sequence_schema : Schema
        The schema representing sequence features

    """

    def __init__(self, context_schema: Schema, sequence_schema: Schema, **kwargs):
        super().__init__(**kwargs)
        self.context_schema = context_schema
        self.sequence_schema = sequence_schema

    def call(self, inputs: TabularData) -> TabularData:
        inputs = self._broadcast(inputs)
        return inputs

    def _check_sequence_features(self, inputs: TabularData):
        sequence_features = self.sequence_schema.column_names

        if len(sequence_features) == 0:
            return

        not_found_seq_features = set(sequence_features).difference(set(inputs.keys()))
        if len(not_found_seq_features) > 0:
            raise ValueError(
                f"Some sequential features were not found in the inputs: {not_found_seq_features}"
            )

        sequence_length = None
        sequence_is_ragged = None
        for k, v in inputs.items():
            if k in sequence_features:
                if isinstance(v, tf.RaggedTensor):
                    if sequence_is_ragged is False:
                        raise ValueError(
                            "Sequential features must all be ragged or all dense, not both."
                        )
                    new_sequence_length = v.row_lengths()
                    sequence_is_ragged = True
                else:
                    if sequence_is_ragged is True:
                        raise ValueError(
                            "Sequential features must all be ragged or all dense, not both."
                        )
                    new_sequence_length = [v.shape[1]]
                    sequence_is_ragged = False

                # check sequences lengths match
                if sequence_length is not None:
                    sequence_lengths_equal = tf.math.reduce_all(
                        tf.equal(new_sequence_length, sequence_length)
                    )
                    tf.Assert(
                        sequence_lengths_equal,
                        [
                            "Sequential features must share the same sequence lengths",
                            (sequence_length, new_sequence_length),
                        ],
                    )
                sequence_length = new_sequence_length

    def _check_context_features(self, inputs: TabularData):
        context_features = self.context_schema.column_names

        if len(context_features) == 0:
            return

        not_found_seq_features = set(context_features).difference(set(inputs.keys()))
        if len(not_found_seq_features) > 0:
            raise ValueError(
                f"Some contextual features were not found in the inputs: {not_found_seq_features}"
            )

        for k in context_features:
            v = inputs[k]
            if not isinstance(v, tf.Tensor):
                raise ValueError(f"A contextual feature ({k}) should be a dense tf.Tensor")

            if len(v.shape) >= 3:
                raise ValueError(
                    f"A contextual feature ({k}) should be a 1D or " "2D tf.Tensor: {v.shape}."
                )

    @tf.function
    def _broadcast(self, inputs):
        self._check_sequence_features(inputs)
        self._check_context_features(inputs)

        sequence_features = self.sequence_schema.column_names
        context_features = self.context_schema.column_names

        if len(sequence_features) == 0 and len(context_features) == 0:
            return inputs

        sequence_features_values = list(
            [inputs[k] for k in sequence_features if inputs[k] is not None]
        )
        if len(sequence_features_values) == 0:
            return inputs
        template_seq_feature_value = sequence_features_values[0]

        non_seq_target = {}
        for fname in context_features:
            if inputs[fname] is None:
                continue

            if isinstance(template_seq_feature_value, tf.RaggedTensor):
                new_value = inputs[fname]
                while len(new_value.shape) < len(template_seq_feature_value.shape):
                    new_value = tf.expand_dims(new_value, 1)

                # Here broadcast the context feature using the same shape
                # of a 3D ragged sequential feature with compatible
                # So that the context feature shape becomes (batch size, seq length, feature dim)
                non_seq_target[fname] = (
                    tf.ones_like(template_seq_feature_value[..., :1], dtype=new_value.dtype)
                    * new_value
                )
            else:
                shape = inputs[fname].shape
                sequence_length = template_seq_feature_value.shape[1]
                target_shape = shape[:1] + [sequence_length] + shape[1:]
                non_seq_target[fname] = tf.broadcast_to(
                    tf.expand_dims(inputs[fname], 1), target_shape
                )
        inputs = {**inputs, **non_seq_target}

        return inputs

    def compute_output_shape(
        self, input_shape: Dict[str, tf.TensorShape]
    ) -> Dict[str, tf.TensorShape]:
        sequence_length = None
        for k in input_shape:
            if k in self.sequence_schema.column_names:
                sequence_length = input_shape[k][1]
                break

        context_shapes = {}
        for k in input_shape:
            if k in self.context_schema.column_names:
                rest_shape = input_shape[k][1:]
                context_shapes[k] = (
                    input_shape[k][:1] + tf.TensorShape([sequence_length]) + rest_shape
                )

        output_shape = {**input_shape, **context_shapes}

        return output_shape

    def compute_mask(self, inputs: TabularData, mask: Optional[TabularData] = None):
        if mask is None:
            return None

        # find the sequence mask
        sequence_mask = None
        for k in mask:
            if mask[k] is not None and k in self.sequence_schema.column_names:
                sequence_mask = mask[k]
                break

        # no sequence mask found
        if sequence_mask is None:
            return mask

        # set the mask value for those that are none
        masks_context = {}
        for k in mask:
            if mask[k] is None and k in self.context_schema.column_names:
                masks_context[k] = sequence_mask

        new_mask = {**mask, **masks_context}

        return new_mask

    def get_config(self):
        config = super().get_config()
        config["context_schema"] = schema_utils.schema_to_tensorflow_metadata_json(
            self.context_schema
        )
        config["sequence_schema"] = schema_utils.schema_to_tensorflow_metadata_json(
            self.sequence_schema
        )
        return config

    @classmethod
    def from_config(cls, config):
        context_schema = schema_utils.tensorflow_metadata_json_to_schema(
            config.pop("context_schema")
        )
        sequence_schema = schema_utils.tensorflow_metadata_json_to_schema(
            config.pop("sequence_schema")
        )
        return cls(context_schema, sequence_schema, **config)


def expected_input_cols_from_schema(schema: Schema, inputs: Optional[TabularData] = None) -> str:
    """Returns the columns/keys that are expected to be
    present in inputs dictionary based on the schema.
    The particular cases are ragged list/multi-hot columns
    which are represented by two keys in the inputs dict,
    suffixed by "__values" and "__offsets".
    For example, the "categories" ragged list feature are
    represented by the Merlin Dataloader as
    "categories__values" and "categories__offsets" keys,
    which are converted by Merlin Models as single
    Ragged/Dense tensor

    Parameters
    ----------
    schema : Schema
        The schema with the expected features
    inputs : Optional[TabularData]
        If provided, checks is ragged list features
        are already converted from (__ragged, __offsets)
        dict representation

    Returns
    -------
    str
        Returns a list with the expected columns in the
        input dictionary
    """
    schema = schema.remove_by_tag(Tags.TARGET)
    expected_features = []
    for name in schema.column_names:
        if (
            schema[name].shape.is_list
            and schema[name].shape.is_ragged
            and (inputs is None or name not in inputs)
        ):
            expected_features.extend([f"{name}__values", f"{name}__offsets"])
        else:
            expected_features.append(name)
    return expected_features
