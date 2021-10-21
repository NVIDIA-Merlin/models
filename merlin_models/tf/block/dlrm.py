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

from typing import Optional

from merlin_standard_lib import Schema, Tag

from ..core import Block, SequentialBlock
from ..features.tabular import TabularFeatures
from ..layers import DotProductInteraction


def DLRMBlock(
    schema: Schema,
    bottom_block: Block,
    top_block: Optional[Block] = None,
    embedding_dim: Optional[int] = None,
) -> SequentialBlock:
    embedding_dim = embedding_dim or bottom_block.layers[-1].units
    input = TabularFeatures.from_schema(schema, embedding_dim_default=embedding_dim)
    routes = {Tag.CONTINUOUS: bottom_block.prepare(aggregation="concat").as_tabular("continuous")}
    dlrm = input.add_routes(routes, add_rest=True, aggregation="stack").add(DotProductInteraction())

    if top_block:
        dlrm = dlrm.add(top_block)

    return dlrm
