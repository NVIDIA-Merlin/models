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
from merlin.models.tf.core.index import IndexBlock, TopKIndexBlock
from merlin.models.tf.core.tabular import AsTabular, Filter, TabularBlock
from merlin.models.tf.core.transformations import (
    AsDenseFeatures,
    AsRaggedFeatures,
    AsSparseFeatures,
    CategoryEncoding,
    ExpandDims,
    HashedCross,
    HashedCrossAll,
    LabelToOneHot,
)

configure_tensorflow()

from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from merlin.models.loader.tf_utils import configure_tensorflow
from merlin.models.tf.blocks.cross import CrossBlock
from merlin.models.tf.blocks.dlrm import DLRMBlock
from merlin.models.tf.blocks.experts import CGCBlock, MMOEBlock, MMOEGate
from merlin.models.tf.blocks.interaction import DotProductInteraction, FMPairwiseInteraction
from merlin.models.tf.blocks.mlp import DenseResidualBlock, MLPBlock
from merlin.models.tf.blocks.optimizer import (
    LazyAdam,
    MultiOptimizer,
    OptimizerBlocks,
    split_embeddings_on_size,
)
from merlin.models.tf.blocks.retrieval.base import DualEncoderBlock, ItemRetrievalScorer
from merlin.models.tf.blocks.retrieval.matrix_factorization import (
    MatrixFactorizationBlock,
    QueryItemIdsEmbeddingsBlock,
)
from merlin.models.tf.blocks.retrieval.two_tower import TwoTowerBlock
from merlin.models.tf.blocks.sampling.base import ItemSampler
from merlin.models.tf.blocks.sampling.cross_batch import PopularityBasedSampler
from merlin.models.tf.blocks.sampling.in_batch import InBatchSampler
from merlin.models.tf.blocks.sampling.queue import FIFOQueue
from merlin.models.tf.core.aggregation import (
    ConcatFeatures,
    ElementwiseSum,
    ElementwiseSumItemMulti,
    StackFeatures,
)
from merlin.models.tf.core.base import (
    Block,
    EmbeddingWithMetadata,
    ModelContext,
    NoOp,
    right_shift_layer,
)
from merlin.models.tf.core.combinators import (
    Cond,
    MapValues,
    ParallelBlock,
    ResidualBlock,
    SequentialBlock,
)
from merlin.models.tf.data_augmentation.noise import StochasticSwapNoise
from merlin.models.tf.dataset import sample_batch
from merlin.models.tf.inputs.base import InputBlock, InputBlockV2
from merlin.models.tf.inputs.continuous import ContinuousFeatures
from merlin.models.tf.inputs.embedding import (
    AverageEmbeddingsByWeightFeature,
    ContinuousEmbedding,
    EmbeddingFeatures,
    EmbeddingOptions,
    Embeddings,
    EmbeddingTable,
    FeatureConfig,
    SequenceEmbeddingFeatures,
    TableConfig,
)
from merlin.models.tf.losses import LossType
from merlin.models.tf.metrics.topk import (
    AvgPrecisionAt,
    MRRAt,
    NDCGAt,
    PrecisionAt,
    RecallAt,
    TopKMetricsAggregator,
)
from merlin.models.tf.models import benchmark
from merlin.models.tf.models.base import BaseModel, Model, RetrievalModel
from merlin.models.tf.models.ranking import DCNModel, DeepFMModel, DLRMModel, WideAndDeepModel
from merlin.models.tf.models.retrieval import (
    MatrixFactorizationModel,
    TwoTowerModel,
    YoutubeDNNRetrievalModel,
)
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.classification import (
    BinaryClassificationTask,
    MultiClassClassificationTask,
)
from merlin.models.tf.prediction_tasks.multi import PredictionTasks
from merlin.models.tf.prediction_tasks.next_item import NextItemPredictionTask
from merlin.models.tf.prediction_tasks.regression import RegressionTask
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.models.tf.predictions.base import PredictionBlock
from merlin.models.tf.predictions.classification import BinaryPrediction, CategoricalPrediction
from merlin.models.tf.predictions.regression import RegressionPrediction
from merlin.models.tf.predictions.sampling.base import Items, ItemSamplerV2
from merlin.models.tf.predictions.sampling.in_batch import InBatchSamplerV2
from merlin.models.tf.predictions.sampling.popularity import PopularityBasedSamplerV2
from merlin.models.tf.utils import repr_utils
from merlin.models.tf.utils.tf_utils import TensorInitializer

ListWrapper.__repr__ = repr_utils.list_wrapper_repr
_DictWrapper.__repr__ = repr_utils.dict_wrapper_repr

Dense.repr_extra = repr_utils.dense_extra_repr
Layer.__rrshift__ = right_shift_layer
Layer.__repr__ = repr_utils.layer_repr
Loss.__repr__ = repr_utils.layer_repr_no_children
Metric.__repr__ = repr_utils.layer_repr_no_children
Optimizer.__repr__ = repr_utils.layer_repr_no_children

__all__ = [
    "Block",
    "Cond",
    "MapValues",
    "ModelContext",
    "SequentialBlock",
    "ResidualBlock",
    "DualEncoderBlock",
    "CrossBlock",
    "DLRMBlock",
    "MLPBlock",
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
    "EmbeddingTable",
    "AverageEmbeddingsByWeightFeature",
    "Embeddings",
    "FeatureConfig",
    "TableConfig",
    "ParallelPredictionBlock",
    "TwoTowerBlock",
    "MatrixFactorizationBlock",
    "QueryItemIdsEmbeddingsBlock",
    "AsDenseFeatures",
    "AsRaggedFeatures",
    "AsSparseFeatures",
    "CategoryEncoding",
    "HashedCross",
    "HashedCrossAll",
    "ElementwiseSum",
    "ElementwiseSumItemMulti",
    "AsTabular",
    "ConcatFeatures",
    "Filter",
    "ParallelBlock",
    "StackFeatures",
    "DotProductInteraction",
    "FMPairwiseInteraction",
    "LabelToOneHot",
    "PredictionBlock",
    "BinaryPrediction",
    "RegressionPrediction",
    "CategoricalPrediction",
    "PredictionTask",
    "BinaryClassificationTask",
    "MultiClassClassificationTask",
    "RegressionTask",
    "MultiOptimizer",
    "LazyAdam",
    "OptimizerBlocks",
    "split_embeddings_on_size",
    "OptimizerBlocks",
    "ItemRetrievalTask",
    "ItemRetrievalScorer",
    "NextItemPredictionTask",
    "NDCGAt",
    "PrecisionAt",
    "MRRAt",
    "AvgPrecisionAt",
    "RecallAt",
    "BaseModel",
    "TopKMetricsAggregator",
    "Model",
    "RetrievalModel",
    "InputBlock",
    "InputBlockV2",
    "PredictionTasks",
    "StochasticSwapNoise",
    "ExpandDims",
    "NoOp",
    "ItemSampler",
    "EmbeddingWithMetadata",
    "InBatchSampler",
    "PopularityBasedSampler",
    "FIFOQueue",
    "YoutubeDNNRetrievalModel",
    "TwoTowerModel",
    "MatrixFactorizationModel",
    "DLRMModel",
    "DCNModel",
    "DeepFMModel",
    "WideAndDeepModel",
    "losses",
    "LossType",
    "sample_batch",
    "TensorInitializer",
]
