#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from merlin.models.torch import schema
from merlin.models.torch.batch import Batch, Sequence
from merlin.models.torch.block import (
    BatchBlock,
    Block,
    ParallelBlock,
    ResidualBlock,
    ShortcutBlock,
    repeat,
    repeat_parallel,
    repeat_parallel_like,
)
from merlin.models.torch.blocks.attention import CrossAttentionBlock
from merlin.models.torch.blocks.dlrm import DLRMBlock
from merlin.models.torch.blocks.experts import CGCBlock, MMOEBlock, PLEBlock
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.functional import map, walk
from merlin.models.torch.inputs.embedding import EmbeddingTable, EmbeddingTables
from merlin.models.torch.inputs.select import SelectFeatures, SelectKeys
from merlin.models.torch.inputs.tabular import TabularInputBlock, stack_context
from merlin.models.torch.models.base import Model, MultiLoader
from merlin.models.torch.models.ranking import DCNModel, DLRMModel
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.outputs.classification import (
    BinaryOutput,
    CategoricalOutput,
    CategoricalTarget,
    EmbeddingTablePrediction,
)
from merlin.models.torch.outputs.regression import RegressionOutput
from merlin.models.torch.outputs.tabular import TabularOutputBlock
from merlin.models.torch.predict import DaskEncoder, DaskPredictor, EncoderBlock
from merlin.models.torch.router import RouterBlock
from merlin.models.torch.transforms.agg import Concat, Stack
from merlin.models.torch.transforms.sequences import BroadcastToSequence, TabularPadding

input_schema = schema.input_schema
output_schema = schema.output_schema
target_schema = schema.target_schema
feature_schema = schema.feature_schema

__all__ = [
    "Batch",
    "BinaryOutput",
    "Block",
    "BatchBlock",
    "DLRMBlock",
    "MLPBlock",
    "Model",
    "MultiLoader",
    "EmbeddingTable",
    "EmbeddingTables",
    "ParallelBlock",
    "MLPBlock",
    "ModelOutput",
    "TabularOutputBlock",
    "ParallelBlock",
    "Sequence",
    "RegressionOutput",
    "ResidualBlock",
    "RouterBlock",
    "SelectKeys",
    "SelectFeatures",
    "ShortcutBlock",
    "TabularInputBlock",
    "Concat",
    "Stack",
    "schema",
    "repeat",
    "repeat_parallel",
    "repeat_parallel_like",
    "CategoricalOutput",
    "CategoricalTarget",
    "EmbeddingTablePrediction",
    "input_schema",
    "output_schema",
    "feature_schema",
    "target_schema",
    "DLRMBlock",
    "DLRMModel",
    "DCNModel",
    "MMOEBlock",
    "PLEBlock",
    "CGCBlock",
    "TabularPadding",
    "BroadcastToSequence",
    "EncoderBlock",
    "DaskEncoder",
    "DaskPredictor",
    "stack_context",
    "CrossAttentionBlock",
    "map",
    "walk",
]
