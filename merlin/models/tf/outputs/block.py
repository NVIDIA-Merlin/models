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

import warnings
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
    task_blocks: Optional[Union[Layer, Dict[str, Layer]]] = None,
) -> Union[ModelOutput, ParallelBlock]:
    """Creates model output(s) based on the columns tagged as target in the schema.
    It outputs either a ModelOutput (e.g. RegressionOutput, BinaryOutput, CategoricalOutput)
    if there is a single target, or a ParallelBlock with multiple ModelOutput if there are
    multiple targets (multi-task learning).

    Simple Usage::
        outputs = OutputBlock(schema)

    Parameters
    ----------
    schema : Schema
        Schema of the input data. This Schema object will be automatically generated using
        [NVTabular](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).
        Next to this, it's also possible to construct it manually.
    model_outputs: Optional[Union[Sequence[ModelOutput], Dict[str, ModelOutput]]]
        Optional dict or list of ModelOutput. If a dict, the keys must be the
        <target_name>/output_type (e.g. "click/binary_output", "rating/regression_output"))
        This method will create ModelOutput only for the tasks not provided in model_outputs.
    pre : Optional[Layer], optional
        Transformation block to apply before the embeddings lookup, by default None
    post : Optional[Layer], optional
        Transformation block to apply after the embeddings lookup, by default None
    task_blocks : Optional[Union[Layer, Dict[str, Layer]]], optional
        Task blocks to be used as task towers. If a single Layer, it is copied to all
        tasks. If a dict, the keys must match the task names
        (e.g. "click/binary_output", rating/regression_output", "item_id/categorical_output").
        You might want to use the task_blocks to create a task-specific tower
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
    bin = _get_col_set_by_tags(targets_schema, [Tags.BINARY_CLASSIFICATION, Tags.BINARY])

    outputs = {}
    if model_outputs is not None:
        if isinstance(model_outputs, dict):
            outputs = model_outputs
        elif isinstance(model_outputs, (tuple, list)):
            outputs = {m.name: m for m in model_outputs}
        elif isinstance(model_outputs, ModelOutput):
            outputs = {model_outputs.name: model_outputs}
        else:
            raise ValueError(
                "If provided model_outputs should be either a dict or list of ModelOutput"
            )

    cols = []
    for col in targets_schema:
        cols.append(col)

        if col.name in con:
            model_output_cls = RegressionOutput
        elif col.name in bin:
            model_output_cls = BinaryOutput
        elif col.name in cat:
            if col.int_domain.max == 1:
                model_output_cls = BinaryOutput
            else:
                model_output_cls = CategoricalOutput

        task_name = model_output_cls.get_task_name(col.name)

        if task_name in outputs:
            output_block = outputs[task_name]
        else:
            # Creates outputs only for the tasks not provided in model_outputs
            output_block = model_output_cls(col)
            outputs[task_name] = output_block

        _set_task_block(outputs[task_name], col.name, task_blocks)

    if len(outputs) == 1:
        return list(outputs.values())[0]

    return ParallelBlock(outputs, pre=pre, post=post, schema=Schema(cols))


def _get_col_set_by_tags(schema: Schema, tags) -> Set[str]:
    return set(schema.select_by_tag(tags).column_names)


def _set_task_block(
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
            task_block = task_blocks
        else:
            raise ValueError("If provided, task_blocks must be either a Layer or Dict[str, Layer]")
    if task_block:
        # Cloning task block, so that it is independent for every tower
        task_block = task_block.from_config(task_block.get_config())
        if output_block.pre is None:
            output_block.pre = task_block
        else:
            output_block.pre = SequentialBlock([output_block.pre, task_block])


@tf.keras.utils.register_keras_serializable(package="merlin_models")
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
    weight_column_name : Optional[str]
        The column name to be used as weight. If should be present
        in the schema either as an input feature (i.e., tagged as
        Tags.CONTINUOUS or Tags.CATEGORICAL) or target feature
        (i.e., tagged as Tags.TARGET). It is optional if
        binary_class_weights is set (assuming the target column
        will be used as weight column in that case).
    binary_class_weights : Optional[Tuple[float, float]], optional
        If provided, it allows setting the weights to which negative (0)
        and positive values (1) of weight column should be converted
        to result the final sample weights, by default None.
        It expects a two elements tuple: (negative_value, positive_value)
    """

    def __init__(
        self,
        weight_column_name: str = None,
        binary_class_weights: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        if binary_class_weights is None and weight_column_name is None:
            raise ValueError(
                "The weight_column_name is required if " "binary_class_weights is not set."
            )

        self.weight_column_name = weight_column_name
        self.binary_class_weights = binary_class_weights

        super().__init__(**kwargs)

    def call(
        self,
        inputs,
        features=None,
        targets=None,
        training=False,
        testing=False,
        target_name=None,
        **kwargs,
    ) -> Union[Prediction, tf.Tensor]:
        if not (training or testing):
            return inputs

        sample_weight = None

        if self.weight_column_name is not None:
            if targets is not None and self.weight_column_name in targets:
                sample_weight = targets[self.weight_column_name]
            elif features is not None and self.weight_column_name in features:
                sample_weight = features[self.weight_column_name]
            else:
                warnings.warn(
                    f"Not able to find the weight_column_name "
                    f"{self.weight_column_name} among neither"
                    "features or targets"
                )

            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, tf.float32)

        # If binary class weights are provided
        if self.binary_class_weights is not None:
            (neg_weight, pos_weight) = self.binary_class_weights

            if self.weight_column_name is None and targets is not None and target_name is not None:
                # If weight column is not provided, assumes the target column should
                # be used
                sample_weight = targets[target_name]

            if sample_weight is not None:
                sample_weight = tf.where(sample_weight == 1, pos_weight, neg_weight)

        if isinstance(inputs, Prediction):
            if inputs.sample_weight is not None:
                # Allows for multiplicative aggregation of sample weights
                # (e.g. on different sampleby sample columns)
                # by cascading multiple ColumnBasedSampleWeight
                sample_weight = tf.multiply(sample_weight, inputs.sample_weight)
                inputs = inputs.copy_with_updates(sample_weight=sample_weight)
            return inputs
        else:
            if target_name and isinstance(targets, dict) and target_name in targets:
                # When there are multiple tasks, targets is a dict and it is necessary to select
                # the corresponding task target to return in Prediction
                targets = targets[target_name]

            return Prediction(inputs, targets, sample_weight=sample_weight)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            weight_column_name=self.weight_column_name,
            binary_class_weights=self.binary_class_weights,
        )
        return config
