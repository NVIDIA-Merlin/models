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
from tensorflow.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from merlin_standard_lib import Schema, Tag

from .. import data
from ..data.synthetic import SyntheticData
from .block.cross import CrossBlock
from .block.dlrm import DLRMBlock
from .block.inputs import ContinuousEmbedding
from .block.mlp import DenseResidualBlock, MLPBlock
from .block.multi_task import CGCBlock, MMOEBlock, MMOEGate
from .block.retrieval import MatrixFactorizationBlock, TwoTowerBlock
from .core import (
    AsTabular,
    Block,
    DualEncoderBlock,
    Filter,
    Model,
    NoOp,
    ParallelBlock,
    ParallelPredictionBlock,
    PredictionTask,
    ResidualBlock,
    SequentialBlock,
    TabularBlock,
    inputs,
    merge,
    prediction_tasks,
    right_shift_layer,
)
from .features.continuous import ContinuousFeatures
from .features.embedding import EmbeddingFeatures, FeatureConfig, TableConfig
from .features.sequence import TabularSequenceFeatures
from .features.tabular import TabularFeatures
from .layers import DotProductInteraction
from .prediction.classification import BinaryClassificationTask, MultiClassClassificationTask
from .prediction.item_prediction import (
    ExtraNegativeSampling,
    InBatchNegativeSampling,
    ItemRetrievalTask,
    SampledItemPredictionTask,
)
from .prediction.ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt

# from .prediction.multi_task import MMOEHead, PLEHead
from .prediction.regression import RegressionTask
from .tabular.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from .tabular.transformations import (
    AsDenseFeatures,
    AsSparseFeatures,
    ExpandDims,
    StochasticSwapNoise,
)
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
    "ContinuousEmbedding",
    "DotProductInteraction",
    "MMOEGate",
    "MMOEBlock",
    "CGCBlock",
    "DenseResidualBlock",
    "TabularBlock",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "FeatureConfig",
    "TableConfig",
    "TabularFeatures",
    "TabularSequenceFeatures",
    "ParallelPredictionBlock",
    "TwoTowerBlock",
    "MatrixFactorizationBlock",
    "AsDenseFeatures",
    "AsSparseFeatures",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "AsTabular",
    "ConcatFeatures",
    "Filter",
    "ParallelBlock",
    "StackFeatures",
    "PredictionTask",
    "BinaryClassificationTask",
    "MultiClassClassificationTask",
    "RegressionTask",
    "SampledItemPredictionTask",
    "InBatchNegativeSampling",
    "ExtraNegativeSampling",
    "ItemRetrievalTask",
    "NDCGAt",
    "AvgPrecisionAt",
    "RecallAt",
    "Model",
    "inputs",
    "prediction_tasks",
    "merge",
    "StochasticSwapNoise",
    "ExpandDims",
    "NoOp",
    "data",
    "SyntheticData",
]
