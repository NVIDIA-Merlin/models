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
from merlin_standard_lib import Schema, Tag
from tensorflow.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from .. import data
from .block.cross import CrossBlock
from .block.dlrm import DLRMBlock
from .block.mlp import DenseResidualBlock, MLPBlock
from .block.multi_task import MMOE, MMOEGate
from .block.retrieval import Retrieval
from .core import (
    AsTabular,
    Block,
    DualEncoderBlock,
    FilterFeatures,
    Head,
    Match,
    Model,
    NoOp,
    ParallelBlock,
    PredictionTask,
    ResidualBlock,
    SequentialBlock,
    TabularBlock,
    right_shift_layer,
)
from .features.continuous import ContinuousFeatures
from .features.embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .features.tabular import TabularFeatures
from .model.multi_task import MMOEHead, PLEHead
from .model.prediction_task import BinaryClassificationTask, RegressionTask
from .tabular.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from .tabular.transformations import AsDenseFeatures, AsSparseFeatures, StochasticSwapNoise
from .utils import repr_utils

Tag.__hash__ = lambda self: hash(str(self))

ListWrapper.__repr__ = repr_utils.list_wrapper_repr
_DictWrapper.__repr__ = repr_utils.dict_wrapper_repr

Dense.repr_extra = repr_utils.dense_extra_repr
Layer.__rrshift__ = right_shift_layer
Layer.__repr__ = repr_utils.layer_repr
Loss.__repr__ = repr_utils.layer_repr_no_children
Metric.__repr__ = repr_utils.layer_repr_no_children
OptimizerV2.__repr__ = repr_utils.layer_repr_no_children

__all__ = [
    "Schema",
    "Tag",
    "Block",
    "SequentialBlock",
    "ResidualBlock",
    "right_shift_layer",
    "DualEncoderBlock",
    "CrossBlock",
    "DLRMBlock",
    "MLPBlock",
    "MMOEGate",
    "MMOE",
    "DenseResidualBlock",
    "TabularBlock",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "Head",
    "MMOEHead",
    "PLEHead",
    "Retrieval",
    "AsDenseFeatures",
    "AsSparseFeatures",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "AsTabular",
    "ConcatFeatures",
    "FilterFeatures",
    "ParallelBlock",
    "StackFeatures",
    "PredictionTask",
    "BinaryClassificationTask",
    "RegressionTask",
    "Model",
    "StochasticSwapNoise",
    "NoOp",
    "Match",
    "data",
]
