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
from .tabular import FilterFeatures, StackFeatures, ConcatFeatures, AsTabular, TabularLayer, AsSparseLayer, AsDenseLayer
from .blocks.base import right_shift_layer, SequentialBlock
from .features.continuous import ContinuousFeatures
from .features.embedding import EmbeddingFeatures, TableConfig, FeatureConfig
from .features.text import TextEmbeddingFeaturesWithTransformers
from .features.tabular import TabularFeatures
from .heads import Head
from .data import DataLoader, DataLoaderValidator
from .blocks.mlp import MLPBlock
from .blocks.with_head import BlockWithHead
from . import repr as _repr

from tensorflow.keras.layers import Layer, Dense
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.losses import Loss
from tensorflow.python.training.tracking.data_structures import _DictWrapper, ListWrapper

ListWrapper.__repr__ = _repr.list_wrapper_repr
_DictWrapper.__repr__ = _repr.dict_wrapper_repr

Dense.extra_repr = _repr.dense_extra_repr
Layer.__rrshift__ = right_shift_layer
Layer.__repr__ = _repr.layer_repr
Loss.__repr__ = _repr.layer_repr_no_children
Metric.__repr__ = _repr.layer_repr_no_children
