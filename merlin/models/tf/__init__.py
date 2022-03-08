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

# flake8: noqa

# Must happen before any importing of tensorflow to curtail mem usage
from merlin.models.loader.tf_utils import configure_tensorflow

configure_tensorflow()

from tensorflow.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

# Must happen before any importing of tensorflow to curtail mem usage
from merlin.models.loader.tf_utils import configure_tensorflow
from merlin.schema import Schema, Tags

from .. import data
from ..data.synthetic import SyntheticData
from . import losses
from .blocks.core.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from .blocks.core.index import IndexBlock, TopKIndexBlock
from .blocks.core.inputs import InputBlock
from .blocks.core.masking import CausalLanguageModeling, MaskedLanguageModeling
from .blocks.core.transformations import (
    AsDenseFeatures,
    AsSparseFeatures,
    ExpandDims,
    LabelToOneHot,
    StochasticSwapNoise,
)
from .blocks.cross import CrossBlock
from .blocks.dlrm import DLRMBlock
from .blocks.experts import CGCBlock, MMOEBlock, MMOEGate
from .blocks.interaction import DotProductInteraction
from .blocks.mlp import DenseResidualBlock, MLPBlock
from .blocks.retrieval.base import ItemRetrievalScorer
from .blocks.retrieval.matrix_factorization import MatrixFactorizationBlock
from .blocks.retrieval.two_tower import TwoTowerBlock
from .blocks.sampling.base import ItemSampler
from .blocks.sampling.cross_batch import (
    CachedCrossBatchSampler,
    CachedUniformSampler,
    PopularityBasedSampler,
)
from .blocks.sampling.in_batch import InBatchSampler
from .blocks.sampling.queue import FIFOQueue
from .core import (
    AsTabular,
    Block,
    BlockContext,
    DualEncoderBlock,
    EmbeddingWithMetadata,
    Filter,
    Model,
    NoOp,
    ParallelBlock,
    ParallelPredictionBlock,
    PredictionTask,
    ResidualBlock,
    RetrievalModel,
    SequentialBlock,
    TabularBlock,
    right_shift_layer,
)
from .features.continuous import ContinuousFeatures
from .features.embedding import (
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    FeatureConfig,
    SequenceEmbeddingFeatures,
    TableConfig,
)
from .losses import LossType
from .metrics.ranking import AvgPrecisionAt, NDCGAt, RecallAt, ranking_metrics
from .models.ranking import DCNModel, DLRMModel
from .models.retrieval import MatrixFactorizationModel, TwoTowerModel, YoutubeDNNRetrievalModel
from .prediction_tasks.classification import BinaryClassificationTask, MultiClassClassificationTask
from .prediction_tasks.multi import PredictionTasks
from .prediction_tasks.next_item import NextItemPredictionTask
from .prediction_tasks.regression import RegressionTask
from .prediction_tasks.retrieval import ItemRetrievalTask
from .utils import repr_utils

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
    "Tags",
    "Block",
    "BlockContext",
    "SequentialBlock",
    "ResidualBlock",
    "right_shift_layer",
    "DualEncoderBlock",
    "CrossBlock",
    "DLRMBlock",
    "MLPBlock",
    "CausalLanguageModeling",
    "MaskedLanguageModeling",
    "ContinuousEmbedding",
    "MMOEGate",
    "MMOEBlock",
    "CGCBlock",
    "TopKIndexBlock",
    "IndexBlock",
    "DenseResidualBlock",
    "TabularBlock",
    "ContinuousFeatures",
    "EmbeddingFeatures",
    "SequenceEmbeddingFeatures",
    "EmbeddingOptions",
    "FeatureConfig",
    "TableConfig",
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
    "ItemRetrievalTask",
    "ItemRetrievalScorer",
    "NextItemPredictionTask",
    "NDCGAt",
    "AvgPrecisionAt",
    "RecallAt",
    "ranking_metrics",
    "Model",
    "RetrievalModel",
    "InputBlock",
    "PredictionTasks",
    "StochasticSwapNoise",
    "LabelToOneHot",
    "ExpandDims",
    "NoOp",
    "data",
    "SyntheticData",
    "ItemSampler",
    "EmbeddingWithMetadata",
    "InBatchSampler",
    "CachedCrossBatchSampler",
    "CachedUniformSampler",
    "PopularityBasedSampler",
    "FIFOQueue",
    "YoutubeDNNRetrievalModel",
    "TwoTowerModel",
    "MatrixFactorizationModel",
    "DLRMModel",
    "DCNModel",
    "losses",
    "LossType",
]
