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
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from merlin.models.tf.blocks.core.combinators import Filter, SequentialBlock, TabularBlock
from merlin.models.tf.blocks.mlp import DenseMaybeLowRank, InitializerType, RegularizerType
from merlin.models.tf.utils.tf_utils import (
    maybe_deserialize_keras_objects,
    maybe_serialize_keras_objects,
)
from merlin.schema import Schema, Tags


def CrossBlock(
    depth: int = 1,
    filter: Optional[Union[Schema, Tags, List[str], Filter]] = None,
    low_rank_dim: Optional[int] = None,
    use_bias: bool = True,
    kernel_initializer: InitializerType = "truncated_normal",
    bias_initializer: InitializerType = "zeros",
    kernel_regularizer: Optional[RegularizerType] = None,
    bias_regularizer: Optional[RegularizerType] = None,
    inputs: Optional[tf.keras.layers.Layer] = None,
    **kwargs,
) -> SequentialBlock:
    """This block provides a way to create high-order feature interactions
       by a number of stacked Cross Layers, from
       DCN V2: Improved Deep & Cross Network [1].
       See Eq. (1) for full-rank and Eq. (2) for low-rank version.


    References
    ----------
    .. [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf


    Parameters
    ----------
    depth : int, optional
        Number of cross-layers to be stacked, by default 1
    filter : Optional[Union[Schema, Tags, List[str], Filter]], optional
        Features filter to be applied on the input, by default None
    low_rank_dim : Optional[int], optional
        If this argument is provided, the weight (`W in R(dxd)`),
        where d is the input features dimension matrix, is factorized in a
        low-rank matrix W = U*V where U and D have (dxr) shape and
        `low_rank_dim = r`, by default None
    use_bias : bool, optional
        Enables or not the bias term, by default True
    kernel_initializer : InitializerType, optional
        Initializer to use on the kernel matrix, by default "truncated_normal"
    bias_initializer : InitializerType, optional
        Initializer to use on the bias vector, by default "zeros"
    kernel_regularizer : Optional[RegularizerType], optional
        Regularizer to use on the kernel matrix, by default None
    bias_regularizer : Optional[RegularizerType], optional
        Regularizer to use on the bias vector, by default None
    inputs : Optional[tf.keras.layers.Layer], optional
        If an `InputBlock` is provided, this block checks if features are
        being aggregated with concat, otherwise it does that,
        as cross blocks need features to be aggregated before, by default None

    Returns
    -------
    SequentialBlock
        A `SequentialBlock` with a number of stacked Cross layers

    Raises
    ------
    ValueError
        Number of cross layers (depth) should be positive
    """

    layers = [inputs, TabularBlock(aggregation="concat")] if inputs else []

    if depth <= 0:
        raise ValueError(f"Number of cross layers (depth) should be positive but is {depth}.")

    for i in range(depth):
        layers.append(
            Cross(
                low_rank_dim=low_rank_dim,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                output_x0=i < depth - 1,
            )
        )

    return SequentialBlock(layers, filter=filter, block_name="CrossBlock", **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class Cross(tf.keras.layers.Layer):
    """Implementation of the Cross Layers from
       DCN V2: Improved Deep & Cross Network [1]_ -
       See Eq. (1) for full-rank and Eq. (2) for low-rank version.

    This layer creates interactions of all input features. When used inside `CrossBlock`,
    stacked `Cross` layers can be used to high-order features interaction.
    The `call` method accepts `inputs` as a tuple of size 2
    tensors. The first input `x0` is the base layer that contains the original
    features (usually the embedding layer); the second input `xi` is the output
    of the previous `Cross` layer in the stack, i.e., the i-th `Cross`
    layer. For the first `Cross` layer in the stack, x0 = xi.
    The output is x_{i+1} = x0 .* ((W * xi + bias * xi) + xi,
    where .* designates elementwise multiplication, W could be a full-rank
    matrix, or a low-rank matrix U*V to reduce the computational cost, and
    diag_scale increases the diagonal of W to improve training stability (
    especially for the low-rank case).

    References
    ----------
    .. [1]. Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and
       practical lessons for web-scale learning to rank systems." Proceedings
       of the Web Conference 2021. 2021. https://arxiv.org/pdf/2008.13535.pdf


    Parameters
    ----------
    low_rank_dim : Optional[int], optional
        If this argument is provided, the weight (`W in R(dxd)`),
        where d is the input features dimension matrix, is factorized in a
        low-rank matrix W = U*V where U and D have (dxr) shape and
        `low_rank_dim = r`, by default None
    use_bias : bool, optional
        Enables or not the bias term, by default True
    kernel_initializer : InitializerType, optional
        Initializer to use on the kernel matrix, by default "truncated_normal"
    bias_initializer : InitializerType, optional
        Initializer to use on the bias vector, by default "zeros"
    kernel_regularizer : Optional[RegularizerType], optional
        Regularizer to use on the kernel matrix, by default None
    bias_regularizer : Optional[RegularizerType], optional
        Regularizer to use on the bias vector, by default None
    output_x0 : bool
        Whether to return a tuple containing the input of the first layer (`x0`),
        which usually represents the input features concatenated, by default False
    """

    def __init__(
        self,
        low_rank_dim: Optional[int] = None,
        use_bias: bool = True,
        kernel_initializer: InitializerType = "truncated_normal",
        bias_initializer: InitializerType = "zeros",
        kernel_regularizer: Optional[RegularizerType] = None,
        bias_regularizer: Optional[RegularizerType] = None,
        output_x0: bool = False,
        **kwargs,
    ):
        dense = kwargs.pop("dense", None)
        if not dense:
            dense = DenseMaybeLowRank(
                low_rank_dim=low_rank_dim,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        super(Cross, self).__init__(**kwargs)
        self.dense = dense
        self.output_x0 = output_x0

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], **kwargs):
        if isinstance(inputs, tuple):
            x0, x = inputs
        else:
            x0 = x = inputs

        self.validate_inputs(x0, x)

        projected = self.dense(x)

        output = x0 * projected + x
        if self.output_x0:
            return x0, output

        return output

    def validate_inputs(self, x0, x):
        tf.assert_equal(
            tf.shape(x0),
            tf.shape(x),
            message="`x0` ({}) and `x` ({}) shapes mismatch!".format(x0.shape, x.shape),
        )

    def get_config(self):
        config = dict()
        config.update(super(Cross, self).get_config())

        return maybe_serialize_keras_objects(self, config, ["dense"])

    @classmethod
    def from_config(cls, config):
        config = maybe_deserialize_keras_objects(config, ["dense"])

        return cls(**config)
