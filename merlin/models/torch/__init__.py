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
from merlin.schema import Schema

from .. import data

# Block related imports
from .block.base import Block, BlockBase, SequentialBlock, build_blocks, right_shift_block
from .block.mlp import MLPBlock

# Features related imports
from .features.continuous import ContinuousFeatures
from .features.embedding import (
    EmbeddingFeatures,
    FeatureConfig,
    SoftEmbedding,
    SoftEmbeddingFeatures,
    TableConfig,
)
from .features.tabular import TabularFeatures

# Model related imports
from .model.base import Head, Model, PredictionTask
from .model.prediction_task import BinaryClassificationTask, RegressionTask

# Tabular related imports
from .tabular.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from .tabular.base import (
    AsTabular,
    FilterFeatures,
    MergeTabular,
    SequentialTabularTransformations,
    TabularAggregation,
    TabularBlock,
    TabularModule,
    TabularTransformation,
)
from .tabular.transformations import StochasticSwapNoise, TabularLayerNorm

__all__ = [
    "Schema",
    "Tag",
    "SequentialBlock",
    "right_shift_block",
    "build_blocks",
    "BlockBase",
    "TabularBlock",
    "Block",
    "MLPBlock",
    "TabularTransformation",
    "SequentialTabularTransformations",
    "TabularAggregation",
    "StochasticSwapNoise",
    "TabularLayerNorm",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "SoftEmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "Head",
    "Model",
    "PredictionTask",
    "AsTabular",
    "ConcatFeatures",
    "FilterFeatures",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "MergeTabular",
    "StackFeatures",
    "BinaryClassificationTask",
    "RegressionTask",
    "TabularModule",
    "SoftEmbedding",
    "data",
]
