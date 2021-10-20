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
from merlin_standard_lib import Schema
from merlin_standard_lib.schema.tag import Tag, TagsType

from ..core import ResidualBlock, SequentialBlock, is_input_block
from .cross import DenseSameDim


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class MLPBlock(SequentialBlock):
    def __init__(
        self,
        dimensions: List[int],
        activation="relu",
        use_bias: bool = True,
        dropout=None,
        normalization=None,
        filter_features=None,
        inputs: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        if inputs and is_input_block(inputs) and not inputs.aggregation:
            inputs.set_aggregation("concat")

        layers = [inputs] if inputs else []

        for dim in dimensions:
            layers.append(tf.keras.layers.Dense(dim, activation=activation, use_bias=use_bias))
            if dropout:
                layers.append(tf.keras.layers.Dropout(dropout))
            if normalization:
                if normalization == "batch_norm":
                    layers.append(tf.keras.layers.BatchNormalization())
                elif isinstance(normalization, tf.keras.layers.Layer):
                    layers.append(normalization)
                else:
                    raise ValueError(
                        "Normalization needs to be an instance `Layer` or " "`batch_norm`"
                    )

        super().__init__(layers, filter_features=filter_features, **kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        dimensions: List[int],
        activation="relu",
        use_bias: bool = True,
        dropout=None,
        normalization=None,
        continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
        **kwargs,
    ) -> "MLPBlock":
        from .. import TabularFeatures

        inputs = TabularFeatures.from_schema(
            schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation="concat",
        )

        return cls(
            dimensions,
            activation=activation,
            use_bias=use_bias,
            dropout=dropout,
            normalization=normalization,
            inputs=inputs,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class MLPResidualBlock(SequentialBlock):
    def __init__(
        self,
        depth: int,
        projection_dim: Optional[int] = None,
        activation="relu",
        use_bias: bool = True,
        dropout=None,
        normalization="batch_norm",
        filter_features=None,
        inputs: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        if inputs and is_input_block(inputs) and not inputs.aggregation:
            inputs.set_aggregation("concat")

        layers = [inputs] if inputs else []

        for d in range(depth):
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
                    raise ValueError(
                        "Normalization needs to be an instance `Layer` or " "`batch_norm`"
                    )

            block = SequentialBlock(block_layers, block_name=f"Dense-{d}", copy_layers=False)
            layers.append(ResidualBlock(block, aggregation=activation))

        super().__init__(layers, filter_features=filter_features, **kwargs)

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        depth: int,
        projection_dim: Optional[int] = None,
        activation="relu",
        use_bias: bool = True,
        dropout=None,
        normalization="batch_norm",
        continuous_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tag]]] = (Tag.CATEGORICAL,),
        **kwargs,
    ) -> "MLPResidualBlock":
        from .. import TabularFeatures

        inputs = TabularFeatures.from_schema(
            schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            aggregation="concat",
        )

        return cls(
            depth=depth,
            projection_dim=projection_dim,
            activation=activation,
            use_bias=use_bias,
            dropout=dropout,
            normalization=normalization,
            inputs=inputs,
            **kwargs,
        )
