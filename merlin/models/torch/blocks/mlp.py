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
from typing import List

import torch


class MLPBlock(torch.nn.Sequential):
    """MLPBlock"""

    def __init__(
        self,
        dimensions: List[int],
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        dropout=None,
        normalization=None,
        no_activation_last_layer: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(dimensions, int):
            dimensions = [dimensions]

        args = []
        for idx, dim in enumerate(dimensions):
            _activation = activation
            if no_activation_last_layer:
                _activation = activation if idx < len(dimensions) - 1 else None
            args.append(
                Dense(dim, activation=_activation, use_bias=use_bias, normalization=normalization)
            )
            if dropout is not None:
                args.append(torch.nn.Dropout(dropout))

        super().__init__(*args)


class Dense(torch.nn.Sequential):
    """Dense implementation, inspired by the one in Keras."""

    def __init__(
        self,
        dim: int,
        activation=torch.nn.ReLU,
        use_bias: bool = True,
        normalization=None,
    ):
        args: List[torch.nn.Module] = [torch.nn.LazyLinear(dim, bias=use_bias)]
        if activation:
            args.append(activation(inplace=True))
        if normalization:
            if normalization == "batch_norm":
                args.append(torch.nn.BatchNorm1d(dim))

        super().__init__(*args)

    def _get_name(self):
        return "Dense"
