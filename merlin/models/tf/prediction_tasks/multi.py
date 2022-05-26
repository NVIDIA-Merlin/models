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
from typing import Dict, Optional, Union

from tensorflow.keras.layers import Layer

from merlin.models.tf.prediction_tasks.base import ParallelPredictionBlock
from merlin.schema import Schema


def PredictionTasks(
    schema: Schema,
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
    task_weight_dict: Optional[Dict[str, float]] = None,
    bias_block: Optional[Layer] = None,
    **kwargs,
) -> ParallelPredictionBlock:
    return ParallelPredictionBlock.from_schema(
        schema,
        task_blocks=task_blocks,
        task_weight_dict=task_weight_dict,
        bias_block=bias_block,
        **kwargs,
    )
