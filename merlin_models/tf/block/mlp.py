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

from typing import List, Optional

import tensorflow as tf

from ..core import ResidualBlock, SequentialBlock
from .cross import DenseSameDim


def MLPBlock(
    dimensions: List[int],
    activation="relu",
    use_bias: bool = True,
    dropout=None,
    normalization=None,
) -> SequentialBlock:
    block_layers = []
    for dim in dimensions:
        block_layers.append(tf.keras.layers.Dense(dim, activation=activation, use_bias=use_bias))
        if dropout:
            block_layers.append(tf.keras.layers.Dropout(dropout))
        if normalization:
            if normalization == "batch_norm":
                block_layers.append(tf.keras.layers.BatchNormalization())
            elif isinstance(normalization, tf.keras.layers.Layer):
                block_layers.append(normalization)
            else:
                raise ValueError("Normalization needs to be an instance `Layer` or " "`batch_norm`")

    return SequentialBlock(block_layers, block_name="MLPBlock")


def DenseResidualBlock(
    projection_dim: Optional[int] = None,
    activation="relu",
    use_bias: bool = True,
    dropout=None,
    normalization="batch_norm",
) -> ResidualBlock:
    block_layers = []
    block_layers.append(DenseSameDim(projection_dim, activation=None, use_bias=use_bias))
    if dropout:
        block_layers.append(tf.keras.layers.Dropout(dropout))
    if normalization:
        if normalization == "batch_norm":
            block_layers.append(tf.keras.layers.BatchNormalization())
        elif isinstance(normalization, tf.keras.layers.Layer):
            block_layers.append(normalization)
        else:
            raise ValueError("Normalization needs to be an instance `Layer` or " "`batch_norm`")

    return ResidualBlock(
        SequentialBlock(block_layers, block_name="DenseResidual"), activation=activation
    )
