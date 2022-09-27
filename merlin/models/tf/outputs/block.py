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

from typing import Set, Union

from merlin.models.tf.core.combinators import ParallelBlock
from merlin.models.tf.outputs.base import ModelOutput
from merlin.models.tf.outputs.classification import BinaryOutput, CategoricalOutput
from merlin.models.tf.outputs.regression import RegressionOutput
from merlin.schema import Schema, Tags


def OutputBlock(
    schema: Schema, pre=None, post=None, **branches
) -> Union[ModelOutput, ParallelBlock]:
    targets_schema = schema.select_by_tag(Tags.TARGET)
    cols = []

    if schema:
        con = _get_col_set_by_tags(targets_schema, [Tags.CONTINUOUS, Tags.REGRESSION])
        cat = _get_col_set_by_tags(
            targets_schema, [Tags.CATEGORICAL, Tags.MULTI_CLASS_CLASSIFICATION]
        )
        bin = _get_col_set_by_tags(targets_schema, [Tags.BINARY_CLASSIFICATION, "binary"])

        outputs = {**branches}

        for col in targets_schema:
            if col.name in branches:
                continue
            if col.name in con:
                outputs[col.name] = RegressionOutput(col)
            elif col.name in bin:
                outputs[col.name] = BinaryOutput(col)
            elif col.name in cat:
                if col.int_domain.max == 1:
                    outputs[col.name] = BinaryOutput(col)
                else:
                    outputs[col.name] = CategoricalOutput(col)

            if col.name in outputs:
                cols.append(col)

    if not outputs:
        raise ValueError(
            "No targets found in schema. Please tag your targets or provide them as branches."
        )

    if len(outputs) == 1:
        return list(outputs.values())[0]

    return ParallelBlock(*list(outputs.values()), pre=pre, post=post, schema=Schema(cols))


def _get_col_set_by_tags(schema: Schema, tags) -> Set[str]:
    return set(schema.select_by_tag(tags).column_names)
