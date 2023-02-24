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
from merlin.dataloader.tf_utils import configure_tensorflow
from merlin.models.tf.core.index import IndexBlock, TopKIndexBlock
from merlin.models.tf.core.tabular import AsTabular, Filter, TabularBlock

configure_tensorflow()

from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from merlin.models.tf.blocks.cross import CrossBlock
from merlin.models.tf.blocks.dlrm import DLRMBlock
from merlin.models.tf.blocks.experts import CGCBlock, ExpertsGate, MMOEBlock, PLEBlock
from merlin.models.tf.blocks.interaction import (
    DotProductInteraction,
    FMBlock,
    FMPairwiseInteraction,
)
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
from merlin.models.tf.core.encoder import EmbeddingEncoder, Encoder, TopKEncoder
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.inputs.base import InputBlock, InputBlockV2
from merlin.models.tf.inputs.continuous import Continuous, ContinuousFeatures, ContinuousProjection
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
from merlin.models.tf.loader import KerasSequenceValidator, Loader, sample_batch
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
from merlin.models.tf.models.base import BaseModel, Model, RetrievalModel, RetrievalModelV2
from merlin.models.tf.models.ranking import DCNModel, DeepFMModel, DLRMModel, WideAndDeepModel
from merlin.models.tf.models.retrieval import (
    MatrixFactorizationModel,
    MatrixFactorizationModelV2,
    TwoTowerModel,
    TwoTowerModelV2,
    YoutubeDNNRetrievalModel,
    YoutubeDNNRetrievalModelV2,
)
from merlin.models.tf.outputs.base import ModelOutput
from merlin.models.tf.outputs.block import ColumnBasedSampleWeight, OutputBlock
from merlin.models.tf.outputs.classification import BinaryOutput, CategoricalOutput
from merlin.models.tf.outputs.contrastive import ContrastiveOutput
from merlin.models.tf.outputs.regression import RegressionOutput
from merlin.models.tf.outputs.sampling.base import Candidate, CandidateSampler
from merlin.models.tf.outputs.sampling.in_batch import InBatchSamplerV2
from merlin.models.tf.outputs.sampling.popularity import PopularityBasedSamplerV2
from merlin.models.tf.outputs.topk import TopKOutput
from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock, PredictionTask
from merlin.models.tf.prediction_tasks.classification import (
    BinaryClassificationTask,
    MultiClassClassificationTask,
)
from merlin.models.tf.prediction_tasks.multi import PredictionTasks
from merlin.models.tf.prediction_tasks.regression import RegressionTask
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.models.utils.dependencies import is_transformers_available

if is_transformers_available():
    from merlin.models.tf.transformers.block import (
        AlbertBlock,
        BertBlock,
        GPT2Block,
        RobertaBlock,
        TransformerBlock,
        XLNetBlock,
    )
    from merlin.models.tf.transformers.transforms import (
        AttentionWeights,
        HiddenStates,
        LastHiddenState,
        LastHiddenStateAndAttention,
    )

from merlin.models.tf.transforms.features import (
    BroadcastToSequence,
    CategoryEncoding,
    HashedCross,
    HashedCrossAll,
    PrepareFeatures,
    ToDense,
    ToOneHot,
    ToSparse,
    ToTarget,
)
from merlin.models.tf.transforms.noise import StochasticSwapNoise
from merlin.models.tf.transforms.regularization import L2Norm
from merlin.models.tf.transforms.sequence import (
    ReplaceMaskedEmbeddings,
    SequenceMaskLast,
    SequenceMaskLastInference,
    SequenceMaskRandom,
    SequencePredictLast,
    SequencePredictNext,
    SequencePredictRandom,
    SequenceTargetAsInput,
)
from merlin.models.tf.transforms.tensor import ExpandDims
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
    "TopKEncoder",
    "Encoder",
    "EmbeddingEncoder",
    "CrossBlock",
    "DLRMBlock",
    "MLPBlock",
    "ContinuousEmbedding",
    "ExpertsGate",
    "MMOEBlock",
    "CGCBlock",
    "PLEBlock",
    "TopKIndexBlock",
    "IndexBlock",
    "DenseResidualBlock",
    "TabularBlock",
    "ContinuousFeatures",
    "Continuous",
    "ContinuousProjection",
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
    "PrepareFeatures",
    "ToSparse",
    "ToDense",
    "ToTarget",
    "CategoryEncoding",
    "BroadcastToSequence",
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
    "FMBlock",
    "ToOneHot",
    "ModelOutput",
    "OutputBlock",
    "ColumnBasedSampleWeight",
    "BinaryOutput",
    "RegressionOutput",
    "CategoricalOutput",
    "ContrastiveOutput",
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
    "NDCGAt",
    "PrecisionAt",
    "MRRAt",
    "AvgPrecisionAt",
    "RecallAt",
    "BaseModel",
    "TopKMetricsAggregator",
    "Model",
    "RetrievalModel",
    "RetrievalModelV2",
    "InputBlock",
    "InputBlockV2",
    "PredictionTasks",
    "StochasticSwapNoise",
    "ExpandDims",
    "L2Norm",
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
    "BroadcastToSequence",
    "Loader",
    "KerasSequenceValidator",
    "SequencePredictNext",
    "SequencePredictLast",
    "SequencePredictRandom",
    "SequenceTargetAsInput",
    "SequenceMaskLast",
    "SequenceMaskRandom",
    "ReplaceMaskedEmbeddings",
    "Prediction",
]
