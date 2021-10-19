from typing import Optional, Tuple, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType

from ..core import SequentialBlock, is_input_block
from ..utils.tf_utils import maybe_serialize_keras_objects

InitializerType = Union[str, tf.keras.initializers.Initializer]
RegularizerType = Union[str, tf.keras.regularizers.Regularizer]


class DenseSameDim(tf.keras.layers.Layer):
    def __init__(
        self,
        projection_dim: Optional[int] = None,
        use_bias: bool = True,
        activation=None,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        last_dim = input_shape[-1]

        dense = tf.keras.layers.Dense(
            last_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
        )

        if self.projection_dim is None:
            self.dense = dense
        else:
            self.dense_u = tf.keras.layers.Dense(
                self.projection_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
            )
            self.dense_v = dense
        super(DenseSameDim, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.projection_dim is None:
            return self.dense(inputs)

        return self.dense_v(self.dense_u(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = dict(
            projection_dim=self.projection_dim, use_bias=self.use_bias, activation=self.activation
        )
        config.update(super(DenseSameDim, self).get_config())

        return maybe_serialize_keras_objects(
            self,
            config,
            ["kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer"],
        )


@tf.keras.utils.register_keras_serializable()
class Cross(tf.keras.layers.Layer):
    def __init__(
        self,
        projection_dim: Optional[int] = None,
        diagonal_scale: Optional[float] = 0.0,
        use_bias: bool = True,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        output_x0: bool = False,
        **kwargs
    ):
        super(Cross, self).__init__(**kwargs)

        self.diagonal_scale = diagonal_scale
        self.output_x0 = output_x0
        self.dense = kwargs.get(
            "dense",
            DenseSameDim(
                projection_dim=projection_dim,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            ),
        )

        self._supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: tf.Tensor, **kwargs):
        if isinstance(inputs, tf.Tensor):
            x, x0 = inputs, inputs
        else:
            x0, x = inputs

        self.validate_inputs(x0, x)

        projected = self.dense(x)

        if self.diagonal_scale:
            projected = projected + self.diagonal_scale * x

        output = x0 * projected + x
        if self.output_x0:
            return x0, output

        return output

    def validate_inputs(self, x0, x):
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError(
                "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
                "dimension {}. This case is not supported yet.".format(x0.shape[-1], x.shape[-1])
            )

    def get_config(self):
        config = dict(diagonal_scale=self.diagonal_scale)
        config.update(super(Cross, self).get_config())

        return maybe_serialize_keras_objects(self, config, ["dense"])


class CrossBlock(SequentialBlock):
    def __init__(
        self,
        depth: int = 1,
        filter_features=None,
        projection_dim: Optional[int] = None,
        diagonal_scale: Optional[float] = 0.0,
        use_bias: bool = True,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        inputs: Optional[tf.keras.layers.Layer] = None,
        **kwargs
    ):
        if inputs and is_input_block(inputs) and not inputs.aggregation:
            inputs.set_aggregation("concat")

        layers = [inputs] if inputs else []

        for i in range(depth):
            layers.append(
                Cross(
                    projection_dim=projection_dim,
                    diagonal_scale=diagonal_scale,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    output_x0=i < depth - 1,
                )
            )

        super().__init__(layers, filter_features=filter_features, **kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        depth: int = 1,
        filter_features=None,
        projection_dim: Optional[int] = None,
        diagonal_scale: Optional[float] = 0.0,
        use_bias: bool = True,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
        **kwargs
    ) -> "CrossBlock":
        from ..features.tabular import TabularFeatures

        inputs = TabularFeatures.from_schema(
            schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation="concat",
        )

        return cls(
            depth=depth,
            filter_features=filter_features,
            projection_dim=projection_dim,
            diagonal_scale=diagonal_scale,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            inputs=inputs,
            **kwargs
        )
