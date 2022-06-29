import abc
import copy
from typing import Dict, List, Optional, Sequence, Union, overload

import tensorflow as tf

from merlin.models.config.schema import SchemaMixin
from merlin.models.tf.blocks.core.base import Block, BlockType, right_shift_layer
from merlin.models.tf.typing import TabularData, TensorOrTabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.registry import Registry, RegistryMixin
from merlin.schema import Schema, Tags

tabular_aggregation_registry: Registry = Registry.class_registry("tf.tabular_aggregations")


class TabularAggregation(
    SchemaMixin, tf.keras.layers.Layer, RegistryMixin["TabularAggregation"], abc.ABC
):
    registry = tabular_aggregation_registry

    """Aggregation of `TabularData` that outputs a single `Tensor`"""

    def call(self, inputs: TabularData, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    def _expand_non_sequential_features(self, inputs: TabularData) -> TabularData:
        inputs_sizes = {k: v.shape for k, v in inputs.items()}
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(inputs_sizes)

        if len(seq_features_shapes) > 0:
            non_seq_features = set(inputs.keys()).difference(set(seq_features_shapes.keys()))
            for fname in non_seq_features:
                # Including the 2nd dim and repeating for the sequence length
                inputs[fname] = tf.tile(tf.expand_dims(inputs[fname], 1), (1, sequence_length, 1))

        return inputs

    def _get_seq_features_shapes(self, inputs_sizes: Dict[str, tf.TensorShape]):
        seq_features_shapes = dict()
        for fname, fshape in inputs_sizes.items():
            # Saves the shapes of sequential features
            if len(fshape) >= 3:
                seq_features_shapes[fname] = tuple(fshape[:2])

        sequence_length = 0
        if len(seq_features_shapes) > 0:
            if len(set(seq_features_shapes.values())) > 1:
                raise ValueError(
                    "All sequential features must share the same shape in the first two dims "
                    "(batch_size, seq_length): {}".format(seq_features_shapes)
                )

            sequence_length = list(seq_features_shapes.values())[0][1]

        return seq_features_shapes, sequence_length

    def _check_concat_shapes(self, inputs: TabularData):
        input_sizes = {k: v.shape for k, v in inputs.items()}
        if len(set([tuple(v[:-1]) for v in input_sizes.values()])) > 1:
            raise Exception(
                "All features dimensions except the last one must match: {}".format(input_sizes)
            )

    def _get_agg_output_size(self, input_size, agg_dim, axis=-1):
        batch_size = tf_utils.calculate_batch_size_from_input_shapes(input_size)
        seq_features_shapes, sequence_length = self._get_seq_features_shapes(input_size)

        if len(seq_features_shapes) > 0:
            return batch_size, sequence_length, agg_dim

        return tf.TensorShape((batch_size, agg_dim))

    def get_values(self, inputs: TabularData) -> List[tf.Tensor]:
        values = []
        for value in inputs.values():
            if type(value) is dict:
                values.extend(self.get_values(value))  # type: ignore
            else:
                values.append(value)

        return values


TabularAggregationType = Union[str, TabularAggregation]

TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `call`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `call`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `call`-method to output a single Tensor.

        Next to providing a class that extends TabularAggregation, it's also possible to provide
        the name that the class is registered in the `tabular_aggregation_registry`. Out of the box
        this contains: "concat", "stack", "element-wise-sum" &
        "element-wise-sum-item-multi".
    schema: Optional[DatasetSchema]
        DatasetSchema containing the columns used in this block.
    name: Optional[str]
        Name of the layer.
"""


@docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING)
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TabularBlock(Block):
    """Layer that's specialized for tabular-data by integrating many often used operations.

    Note, when extending this class, typically you want to overwrite the `compute_call_output_shape`
    method instead of the normal `compute_output_shape`. This because a Block can contain pre- and
    post-processing and the output-shapes are handled automatically in `compute_output_shape`. The
    output of `compute_call_output_shape` should be the shape that's outputted by the `call`-method.

    Parameters
    ----------
    {tabular_module_parameters}
    """

    def __init__(
        self,
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        is_input: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_size = None
        self.set_pre(pre)
        self.set_post(post)
        self.set_aggregation(aggregation)
        self._is_input = is_input

        if schema:
            self.set_schema(schema)

    @property
    def is_input(self) -> bool:
        return self._is_input

    @classmethod
    def from_schema(
        cls, schema: Schema, tags=None, allow_none=True, **kwargs
    ) -> Optional["TabularBlock"]:
        """Instantiate a TabularLayer instance from a DatasetSchema.

        Parameters
        ----------
        schema
        tags
        kwargs

        Returns
        -------
        Optional[TabularModule]
        """
        schema_copy = copy.copy(schema)
        if tags:
            schema_copy = schema_copy.select_by_tag(tags)
            if not schema_copy.column_names and not allow_none:
                raise ValueError(f"No features with tags: {tags} found")

        if not schema_copy.column_names:
            return None

        return cls.from_features(schema_copy.column_names, schema=schema_copy, **kwargs)

    @classmethod
    @docstring_parameter(tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING, extra_padding=4)
    def from_features(
        cls,
        features: List[str],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        name=None,
        **kwargs,
    ) -> "TabularBlock":
        """
        Initializes a TabularLayer instance where the contents of features will be filtered out

        Parameters
        ----------
        features: List[str]
            A list of feature-names that will be used as the first pre-processing op to filter out
            all other features not in this list.
        {tabular_module_parameters}

        Returns
        -------
        TabularModule
        """
        pre = [Filter(features), pre] if pre else Filter(features)  # type: ignore

        return cls(pre=pre, post=post, aggregation=aggregation, name=name, **kwargs)

    def pre_call(
        self, inputs: TabularData, transformations: Optional[BlockType] = None
    ) -> TabularData:
        """Method that's typically called before the forward method for pre-processing.

        Parameters
        ----------
        inputs: TabularData
             input-data, typically the output of the forward method.
        transformations: TabularTransformationsType, optional

        Returns
        -------
        TabularData
        """
        return self._maybe_apply_transformations(
            inputs, transformations=transformations or self.pre
        )

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        return inputs

    def post_call(
        self,
        inputs: TabularData,
        transformations: Optional[BlockType] = None,
        merge_with: Optional[Union["TabularBlock", List["TabularBlock"]]] = None,
        aggregation: Optional[TabularAggregationType] = None,
    ) -> TensorOrTabularData:
        """Method that's typically called after the forward method for post-processing.

        Parameters
        ----------
        inputs: TabularData
            input-data, typically the output of the forward method.
        transformations: TabularTransformationType, optional
            Transformations to apply on the input data.
        merge_with: Union[TabularModule, List[TabularModule]], optional
            Other TabularModule's to call and merge the outputs with.
        aggregation: TabularAggregationType, optional
            Aggregation to aggregate the output to a single Tensor.

        Returns
        -------
        TensorOrTabularData (Tensor when aggregation is set, else TabularData)
        """
        _aggregation: Optional[TabularAggregation] = None
        if aggregation:
            _aggregation = TabularAggregation.parse(aggregation)
        _aggregation = _aggregation or getattr(self, "aggregation", None)

        outputs = inputs
        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        outputs = self._maybe_apply_transformations(
            outputs, transformations=transformations or self.post
        )

        if _aggregation:
            if self.has_schema:
                _aggregation.set_schema(self.schema)
            return _aggregation(outputs)

        return outputs

    def _maybe_apply_transformations(
        self,
        inputs: TabularData,
        transformations: Optional[BlockType] = None,
    ) -> TabularData:
        """Apply transformations to the inputs if these are defined.

        Parameters
        ----------
        inputs
        transformations

        Returns
        -------

        """
        if transformations:
            _transformations = Block.parse(transformations)
            return _transformations(inputs)

        return inputs

    def compute_call_output_shape(self, input_shapes):
        return input_shapes

    def compute_output_shape(self, input_shapes):
        if self.pre:
            input_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self._check_post_output_size(self.compute_call_output_shape(input_shapes))

        return output_shapes

    def build(self, input_shapes):
        super().build(input_shapes)
        output_shapes = input_shapes
        if self.pre:
            self.pre.build(input_shapes)
            output_shapes = self.pre.compute_output_shape(input_shapes)

        output_shapes = self.compute_call_output_shape(output_shapes)

        if isinstance(output_shapes, dict):
            if self.post:
                self.post.build(output_shapes)
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
                if self.has_schema:
                    self.aggregation.set_schema(self.schema)

                self.aggregation.build(output_shapes)

    def get_config(self):
        config = super(TabularBlock, self).get_config()
        config = tf_utils.maybe_serialize_keras_objects(
            self, config, ["pre", "post", "aggregation"]
        )

        if self.has_schema:
            config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)

        return config

    @property
    def is_tabular(self) -> bool:
        return True

    def __add__(self, other):
        from models.tf.blocks.core.combinators import ParallelBlock

        return ParallelBlock(self, other)

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        return super().from_config(config)

    def _check_post_output_size(self, input_shapes):
        output_shapes = input_shapes

        if isinstance(output_shapes, dict):
            if self.post:
                output_shapes = self.post.compute_output_shape(output_shapes)
            if self.aggregation:
                schema = getattr(self, "_schema", None)
                self.aggregation.set_schema(schema)
                output_shapes = self.aggregation.compute_output_shape(output_shapes)

        return output_shapes

    def apply_to_all(self, inputs, columns_to_filter=None):
        if columns_to_filter:
            inputs = Filter(columns_to_filter)(inputs)
        outputs = tf.nest.map_structure(self, inputs)

        return outputs

    def set_schema(self, schema=None):
        self._maybe_set_schema(self.pre, schema)
        self._maybe_set_schema(self.post, schema)
        self._maybe_set_schema(self.aggregation, schema)

        return super().set_schema(schema)

    def set_pre(self, value: Optional[BlockType]):
        self._pre = Block.parse(value) if value else None

    @property
    def pre(self) -> Optional[Block]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._pre

    @property
    def post(self) -> Optional[Block]:
        """

        Returns
        -------
        SequentialTabularTransformations, optional
        """
        return self._post

    def set_post(self, value: Optional[BlockType]):
        self._post = Block.parse(value) if value else None

    @property
    def aggregation(self) -> Optional[TabularAggregation]:
        """

        Returns
        -------
        TabularAggregation, optional
        """
        return self._aggregation

    def set_aggregation(self, value: Optional[Union[str, TabularAggregation]]):
        """

        Parameters
        ----------
        value
        """
        if value:
            self._aggregation: Optional[TabularAggregation] = TabularAggregation.parse(value)
        else:
            self._aggregation = None

    def repr_ignore(self):
        return []

    def repr_extra(self):
        return []

    def repr_add(self):
        return []

    @staticmethod
    def calculate_batch_size_from_input_shapes(input_shapes):
        return tf_utils.calculate_batch_size_from_input_shapes(input_shapes)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def super(self):
        return super()


# This is done like this to avoid mypy crashing.
def _tabular_call(  # type: ignore
    self,
    inputs: TabularData,
    *args,
    pre: Optional[BlockType] = None,
    post: Optional[BlockType] = None,
    merge_with: Optional[Union["TabularBlock", List["TabularBlock"]]] = None,
    aggregation: Optional[TabularAggregationType] = None,
    **kwargs,
) -> TensorOrTabularData:
    """We overwrite the call method in order to be able to do pre- and post-processing.

    Parameters
    ----------
    inputs: TabularData
        Input TabularData.
    pre: TabularTransformationsType, optional
        Transformations to apply before calling the forward method. If pre is None, this method
        will check if `self.pre` is set.
    post: TabularTransformationsType, optional
        Transformations to apply after calling the forward method. If post is None, this method
        will check if `self.post` is set.
    merge_with: Union[TabularModule, List[TabularModule]]
        Other TabularModule's to call and merge the outputs with.
    aggregation: TabularAggregationType, optional
        Aggregation to aggregate the output to a single Tensor.

    Returns
    -------
    TensorOrTabularData (Tensor when aggregation is set, else TabularData)
    """
    inputs = self.pre_call(inputs, transformations=pre)

    # This will call the `call` method implemented by the super class.
    outputs = self.super().__call__(inputs, *args, **kwargs)  # type: ignore

    if isinstance(outputs, dict):
        outputs = self.post_call(
            outputs, transformations=post, merge_with=merge_with, aggregation=aggregation
        )

    return outputs


TabularBlock.__call__ = _tabular_call  # type: ignore


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Filter(TabularBlock):
    """Transformation that filters out certain features from `TabularData`."

    Parameters
    ----------
    to_include: List[str]
        List of features to include in the result of calling the module
    pop: bool
        Boolean indicating whether to pop the features to exclude from the inputs dictionary.
    """

    @overload
    def __init__(
        self,
        inputs: Sequence[str],
        name=None,
        pop=False,
        exclude=False,
        add_to_context: bool = False,
        **kwargs,
    ):
        ...

    @overload
    def __init__(
        self,
        inputs: Union[Schema, Tags],
        name=None,
        pop=False,
        exclude=False,
        add_to_context: bool = False,
        **kwargs,
    ):
        ...

    def __init__(
        self, inputs, name=None, pop=False, exclude=False, add_to_context: bool = False, **kwargs
    ):
        if isinstance(inputs, Tags):
            self.feature_names = inputs
        else:
            self.feature_names = list(inputs.column_names) if isinstance(inputs, Schema) else inputs
        super().__init__(name=name, **kwargs)
        self.exclude = exclude
        self.pop = pop
        self.add_to_context = add_to_context

    def set_schema(self, schema=None):
        out = super().set_schema(schema)

        if isinstance(self.feature_names, Tags):
            self.feature_names = self.schema.select_by_tag(self.feature_names).column_names

        return out

    def call(self, inputs: TabularData, **kwargs) -> TabularData:
        """Filter out features from inputs.

        Parameters
        ----------
        inputs: TabularData
            Input dictionary containing features to filter.

        Returns Filtered TabularData that only contains the feature-names in `self.to_include`.
        -------

        """
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if self.check_feature(k)}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        if self.add_to_context:
            self.context.tensors.update(outputs)

            return {}

        return outputs

    def compute_call_output_shape(self, input_shape):
        if self.add_to_context:
            return {}

        outputs = {k: v for k, v in input_shape.items() if self.check_feature(k)}

        return outputs

    def check_feature(self, feature_name) -> bool:
        if self.exclude:
            return feature_name not in self.feature_names

        return feature_name in self.feature_names

    def get_config(self):
        config = super().get_config()
        config["inputs"] = self.feature_names
        config["exclude"] = self.exclude
        config["pop"] = self.pop

        return config


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AsTabular(tf.keras.layers.Layer):
    """Converts a Tensor to TabularData by converting it to a dictionary.

    Parameters
    ----------
    output_name: str
        Name that should be used as the key in the output dictionary.
    name: str
        Name of the layer.
    """

    def __init__(self, output_name: str, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_name = output_name

    def call(self, inputs, **kwargs):
        return {self.output_name: inputs}

    def compute_output_shape(self, input_shape):
        return {self.output_name: input_shape}

    def get_config(self):
        config = super(AsTabular, self).get_config()
        config["output_name"] = self.output_name

        return config

    @property
    def is_tabular(self) -> bool:
        return True
