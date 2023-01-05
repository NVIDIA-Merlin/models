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

from typing import Dict, Optional, Sequence, Set, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from merlin.models.tf.core.base import Block
from merlin.models.tf.core.combinators import ParallelBlock, SequentialBlock
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.outputs.base import ModelOutput
from merlin.models.tf.outputs.classification import BinaryOutput, CategoricalOutput
from merlin.models.tf.outputs.regression import RegressionOutput
from merlin.schema import Schema, Tags


def OutputBlock(
    schema: Schema,
    model_outputs: Optional[Union[Sequence[ModelOutput], Dict[str, ModelOutput]]] = None,
    pre: Optional[Layer] = None,
    post: Optional[Layer] = None,
    task_pre_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
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
    task_pre_blocks : Optional[Union[Layer, Dict[str, Layer]]], optional
        Task blocks to be used as task towers. If a single Layer, it is copied to all
        tasks. If a dict, the keys must match the task names
        (e.g. "click/binary_output", rating/regression_output", "item_id/categorical_output").
        You might want to use the task_pre_blocks to create a task-specific tower
        (e.g. MLPBLock([32])) or to customize inputs, targets or sample_weights for a
        given task.

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

        _set_task_pre_block(output_block, col.name, task_pre_blocks)
        outputs[task_name] = output_block

    if len(outputs) == 1:
        return list(outputs.values())[0]

    return ParallelBlock(outputs, pre=pre, post=post, schema=Schema(cols))


def _get_col_set_by_tags(schema: Schema, tags) -> Set[str]:
    return set(schema.select_by_tag(tags).column_names)


def _set_task_pre_block(
    output_block: OutputBlock,
    col_name: str,
    task_pre_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
):
    task_block = None
    if task_pre_blocks is not None:
        if isinstance(task_pre_blocks, dict):
            if output_block.name in task_pre_blocks:
                task_block = task_pre_blocks[output_block.name]
            elif col_name in task_pre_blocks:
                task_block = task_pre_blocks[col_name]
        elif isinstance(task_pre_blocks, Layer):
            # Cloning the layer for every task
            task_block = task_pre_blocks.from_config(task_pre_blocks.get_config())
        else:
            raise ValueError("If provided, task_blocks must be either a Layer or Dict[str, Layer]")
    if task_block:
        if output_block.pre is None:
            output_block.pre = task_block
        else:
            output_block.pre = SequentialBlock([output_block.pre, task_block])


class ColumnBasedSampleWeight(Block):
    """Allows using columns (features or targets) as sample weights
    for a give ModelOutput.

    Examples
    ----------
      It can be used for example for binary class weights, using the same
      column as the weight column and setting binary_class_weights.

      ```
      inputs = mm.InputBlockV2(music_streaming_data.schema)
      output_block = mm.BinaryOutput("like",
            post=mm.ColumnBasedSampleWeight(
                weight_column_name="like", binary_class_weights=((1.0, 5.0)
            )
        )
      model = mm.Model(inputs, mm.MLPBlock([64]), output_block)
      ```

      Another use case is computing a loss only for a subset of the examples.
      That is useful in multi-task learning, where one of target is conditioned
      on the other target (e.g. the user can only like if he viewed the video).
      So you can use the positive views (view==1)as the sample space for training the
      "like" prediction task.

      ```
      inputs = mm.InputBlockV2(music_streaming_data.schema)
      output_block = mm.ParallelBlock(
            "view/binary_output": mm.BinaryOutput("view"),
            "like/binary_output": mm.BinaryOutput("like",
                                           post=mm.ColumnBasedSampleWeight(
                                                  weight_column_name="view",
                                        )
                                   )
            )
      model = mm.Model(inputs, mm.MLPBlock([64]), output_block)
      ```

    Parameters
    ----------
    weight_column_name : str
        The column name to be used as weight. If should be present
        in the schema either as an input feature (i.e., tagged as
        Tags.CONTINUOUS or Tags.CATEGORICAL) or target feature
        (i.e., tagged as Tags.TARGET)
    binary_class_weights : Optional[Tuple[float, float]], optional
        If provided, it allows setting the weights to which negative (0)
        and positive values (1) of weight column should be converted
        to result the final sample weights, by default None.
        It expects a two elements tuple: (negative_value, positive_value)
    """

    def __init__(
        self,
        weight_column_name: str,
        binary_class_weights: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        self.weight_column_name = weight_column_name
        self.binary_class_weights = binary_class_weights
        super().__init__(**kwargs)

    def call(self, inputs, features=None, targets=None, target_name=None, **kwargs) -> Prediction:
        sample_weight = None
        if targets is not None and self.weight_column_name in targets:
            sample_weight = targets[self.weight_column_name]
        elif features is not None and self.weight_column_name in features:
            sample_weight = features[self.weight_column_name]
        else:
            raise ValueError(
                f"Not able to find the weight_column_name"
                f"{self.weight_column_name} among "
                "features and targets"
            )

        sample_weight = tf.cast(sample_weight, tf.float32)

        # If the sample weight is a binary column
        if self.binary_class_weights is not None:
            (neg_weight, pos_weight) = self.binary_class_weights
            sample_weight = tf.where(
                sample_weight == 1,
                pos_weight,
                neg_weight,
            )

        if target_name and isinstance(targets, dict) and target_name in targets:
            # When there are multiple tasks, targets is a dict and it is necessary to select
            # the corresponding task target to return in Prediction
            targets = targets[target_name]

        return Prediction(inputs, targets, sample_weight=sample_weight)

    def compute_output_shape(self, input_shape):
        return input_shape
