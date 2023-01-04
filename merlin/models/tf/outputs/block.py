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

from typing import Dict, Optional, Sequence, Set, Union

from tensorflow.keras.layers import Layer

from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock
from merlin.models.tf.outputs.base import ModelOutput
from merlin.models.tf.outputs.classification import BinaryOutput, CategoricalOutput
from merlin.models.tf.outputs.regression import RegressionOutput
from merlin.schema import Schema, Tags


def OutputBlock(
    schema: Schema,
    model_outputs: Optional[Union[Sequence[ModelOutput], Dict[str, ModelOutput]]] = None,
    pre: Optional[Layer] = None,
    post: Optional[Layer] = None,
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
) -> Union[ModelOutput, ParallelBlock]:
    """Creates model output(s) based on the columns tagged as target in the schema.

    Simple Usage::
        outputs = OutputBlock(schema)

    Parameters
    ----------
    schema : Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    pre : Optional[Layer], optional
        Transformation block to apply before the embeddings lookup, by default None
    post : Optional[Layer], optional
        Transformation block to apply after the embeddings lookup, by default None
    task_blocks : Optional[Union[Layer, Dict[str, Layer]]], optional
        Task blocks to be used as task towers. If a single Layer, it is copied to all
        tasks. If a dict, the keys must match the task names
        (e.g. "click/binary_output", rating/regression_output", "item_id/categorical_output").

    Raises
    -------
        ValueError: when the schema does not contain any target columns.

    Returns
    -------
    Union[ModelOutput, ParallelBlock]
        Returns a single output block or a parallel block if there is more than one target column.
    """

    targets_schema = schema.select_by_tag(Tags.TARGET)
    if len(targets_schema) == 0:
        raise ValueError(
            "No targets found in schema. Please tag your targets or provide them as branches."
        )

    con = _get_col_set_by_tags(targets_schema, [Tags.CONTINUOUS, Tags.REGRESSION])
    cat = _get_col_set_by_tags(targets_schema, [Tags.CATEGORICAL, Tags.MULTI_CLASS_CLASSIFICATION])
    bin = _get_col_set_by_tags(targets_schema, [Tags.BINARY_CLASSIFICATION, "binary"])

    outputs = {}
    if model_outputs is not None:
        if isinstance(model_outputs, dict):
            outputs = model_outputs
        elif isinstance(model_outputs, (tuple, list)):
            outputs = {m.name: m for m in model_outputs}
        else:
            raise ValueError(
                "If provided model_outputs should be either a dict or list of ModelOutput"
            )

    cols = []
    for col in targets_schema:
        cols.append(col)
        if col.name in con:
            output_block = RegressionOutput(col)
        elif col.name in bin:
            output_block = BinaryOutput(col)
        elif col.name in cat:
            if col.int_domain.max == 1:
                output_block = BinaryOutput(col)
            else:
                output_block = CategoricalOutput(col)

        task_name = output_block.name
        if task_name in outputs:
            # If this model output is already provided in model_outputs,
            # use that instead of creating a new one for this column
            output_block = outputs[task_name]

        output_block = _add_output_task_block(output_block, col.name, task_blocks)
        outputs[task_name] = output_block

    if len(outputs) == 1:
        return list(outputs.values())[0]

    return ParallelBlock(outputs, pre=pre, post=post, schema=Schema(cols))


def _get_col_set_by_tags(schema: Schema, tags) -> Set[str]:
    return set(schema.select_by_tag(tags).column_names)


def _add_output_task_block(
    output_block: OutputBlock,
    col_name: str,
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
):
    task_block = None
    if task_blocks is not None:
        if isinstance(task_blocks, dict):
            if output_block.name in task_blocks:
                task_block = task_blocks[output_block.name]
            elif col_name in task_blocks:
                task_block = task_blocks[col_name]
        elif isinstance(task_blocks, Layer):
            # Cloning the layer for every task
            task_block = task_blocks.from_config(task_blocks.get_config())
        else:
            raise ValueError("If provided, task_blocks must be either a Layer or Dict[str, Layer]")

    if task_block:
        return SequentialBlock([task_block, output_block])
    else:
        return output_block
