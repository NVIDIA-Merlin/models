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

from merlin.models.tf.losses.base import LossType, loss_registry
from merlin.models.tf.losses.listwise import CategoricalCrossEntropy, SparseCategoricalCrossEntropy
from merlin.models.tf.losses.pairwise import (
    BPRLoss,
    BPRmaxLoss,
    HingeLoss,
    LogisticLoss,
    TOP1Loss,
    TOP1maxLoss,
    TOP1v2Loss,
)

__all__ = [
    "CategoricalCrossEntropy",
    "SparseCategoricalCrossEntropy",
    "BPRLoss",
    "BPRmaxLoss",
    "HingeLoss",
    "LogisticLoss",
    "TOP1Loss",
    "TOP1maxLoss",
    "TOP1v2Loss",
    "loss_registry",
    "LossType",
]
