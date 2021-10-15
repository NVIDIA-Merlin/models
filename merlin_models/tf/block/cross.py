from typing import Optional, Tuple, Union

import tensorflow as tf
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.schema.tag import TagsType

from merlin_models.tf import SequentialBlock
from merlin_models.tf.utils.tf_utils import maybe_serialize_keras_objects

InitializerType = Union[str, tf.keras.initializers.Initializer]
RegularizerType = Union[str, tf.keras.regularizers.Regularizer]


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

        self.projection_dim = projection_dim
        self.diagonal_scale = diagonal_scale
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.input_dim = None
        self.output_x0 = output_x0

        self._supports_masking = True

    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self.projection_dim is None:
            self.dense = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
            )
        else:
            self.dense_u = tf.keras.layers.Dense(
                self.projection_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
            )
            self.dense_v = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
            )
        super(Cross, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: tf.Tensor, **kwargs):
        if isinstance(inputs, tf.Tensor):
            x, x0 = inputs, inputs
        else:
            x0, x = inputs

        self.validate_inputs(x0, x)

        projected = self.project(x)

        if self.diagonal_scale:
            projected = projected + self.diagonal_scale * x

        output = x0 * projected + x
        if self.output_x0:
            return x0, output

        return output

    def project(self, x):
        if self.projection_dim is None:
            return self.dense(x)

        return self.dense_v(self.dense_u(x))

    def validate_inputs(self, x0, x):
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError(
                "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
                "dimension {}. This case is not supported yet.".format(x0.shape[-1], x.shape[-1])
            )

    def get_config(self):
        config = super(Cross, self).get_config()
        config.update(
            dict(
                projection_dim=self.projection_dim,
                diagonal_scale=self.diagonal_scale,
                use_bias=self.use_bias,
            )
        )

        return maybe_serialize_keras_objects(
            self,
            config,
            ["kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer"],
        )


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
        from merlin_models.tf.features.base import is_input_block

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
                    output_x0=i < depth - 1
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
