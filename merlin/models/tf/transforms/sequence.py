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
from typing import Optional, Tuple, Union

import tensorflow as tf

from merlin.models.tf.core.base import Block, BlockType, PredictionOutput
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.transforms.tensor import ListToRagged
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.schema import ColumnSchema, Schema, Tags


@Block.registry.register_with_multiple_names("remove_pad_3d")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RemovePad3D(Block):
    """
    Flatten the sequence of labels and filter out non-targets positions

    Parameters
    ----------
        padding_idx: int
            The padding index value.
            Defaults to 0.
    Returns
    -------
        targets: tf.Tensor
            The flattened vector of true targets positions
        flatten_predictions: tf.Tensor
            If the predictions are 3-D vectors (sequential task),
            flatten the predictions vectors to keep only the ones related to target positions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        targets = tf.reshape(targets, (-1,))
        non_pad_mask = targets != self.padding_idx
        targets = tf.boolean_mask(targets, non_pad_mask)

        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"

        if len(tuple(predictions.get_shape())) == 3:
            predictions = tf.reshape(predictions, (-1, predictions.shape[-1]))
            predictions = tf.boolean_mask(
                predictions, tf.broadcast_to(tf.expand_dims(non_pad_mask, 1), tf.shape(predictions))
            )

        return outputs.copy_with_updates(predictions=predictions, targets=targets)


@Block.registry.register_with_multiple_names("predict_next")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class PredictNext(TabularBlock):
    """Prepares sequential inputs and targets for next-item prediction.
    The target is extracted from the shifted sequence of item ids and
    the sequential input features are truncated in the last position.

    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        pre: Optional[BlockType] = None,
        **kwargs,
    ):
        if not pre:
            pre = ListToRagged()
        super().__init__(pre=pre, schema=schema, **kwargs)

        self.target = target
        self.target_name = self._get_target(target)

    def _get_target(self, target):
        if (
            (isinstance(target, str) and target not in self.schema.column_names)
            or (isinstance(target, Tags) and len(self.schema.select_by_tag(target)) > 0)
            or (isinstance(target, ColumnSchema) and target not in self.schema)
        ):
            raise ValueError("The target column needs to be part of the sequential schema")

        target_name = target
        if isinstance(target, ColumnSchema):
            target_name = target.name
        if isinstance(target, Tags):
            if len(self.schema.select_by_tag(target)) > 1:
                raise ValueError(
                    "Only 1 column should the Tag ({target}) provided for target, but"
                    f"the following columns have that tag: "
                    f"{self.schema.select_by_tag(target).column_names}"
                )
            target_name = self.schema.select_by_tag(target).column_names[0]
        return target_name

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Union[Prediction, Tuple]:
        target_shape = inputs[self.target_name].get_shape().as_list()
        if len(target_shape) != 2:
            raise ValueError(
                f"The target column ({self.target_name}) is expected to be a 2D tensor,"
                f" but the shape is {target_shape}"
            )
        if target_shape[-1] == 1:
            raise ValueError(
                "The 2nd dim of the target column ({self.target_name}) should be greater"
                " than 1, so that the sequential input can be shifted as target"
            )

        seq_inputs_shapes = {
            col: inputs[col].get_shape().as_list() for col in self.schema.column_names
        }

        seq_shapes = list(seq_inputs_shapes.values())
        if not all(x == seq_shapes[0] for x in seq_shapes):
            raise ValueError(
                "The sequential inputs must have the same shape, but the shapes"
                f"are different: {seq_inputs_shapes}"
            )

        # Shifts the target column to be the next item of corresponding input column
        new_target = inputs[self.target_name][:, 1:]
        if targets is None:
            targets = new_target
        elif isinstance(targets, dict):
            targets[self.target_name] = new_target
        else:
            targets = dict()
            targets[self.target_name] = new_target

        new_inputs = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                # Removes the last item of the sequence, as it belongs to the target
                new_inputs[k] = v[:, :-1]
            else:
                new_inputs[k] = v

        return (new_inputs, targets)

    def compute_output_shape(self, input_shape):
        new_input_shapes = dict()
        for k, v in input_shape.items():
            new_input_shapes[k] = v
            if k in self.schema.column_names:
                # If it is a list/sparse feature (in tuple representation), uses the offset as shape
                if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                    new_input_shapes[k] = tf.TensorShape([v[1][0], None])
                else:
                    # Reducing 1 position of the seq length
                    new_input_shapes[k] = tf.TensorShape([v[0], v[1] - 1])

        return new_input_shapes

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        output_column_schemas = {}
        for col_name, col_schema in input_schema.column_schemas.items():
            if col_name in self.schema.column_names:
                # TODO: Fix ColumnSchema.with_properties() to not override the new values
                # of existing properties with old values
                # This should work instead:
                # col_schema.with_properties({"value_count":
                #           Domain(min=col_schema.value_count.min,
                #                  max=col_schema.value_count.max-1,
                #                  name=col_schema.name)})
                output_column_schemas[col_name] = ColumnSchema(
                    col_schema.name,
                    tags=col_schema.tags,
                    properties={
                        **col_schema.properties,
                        **{
                            "value_count": {
                                "min": col_schema.value_count.min,
                                "max": col_schema.value_count.max - 1,
                                "name": col_schema.name,
                            }
                        },
                    },
                    dtype=col_schema.dtype,
                    is_list=col_schema.is_list,
                    is_ragged=col_schema.is_ragged,
                )

            else:
                output_column_schemas[col_name] = col_schema
        return Schema(column_schemas=output_column_schemas)

    def get_config(self):
        """Returns the config of the layer as a Python dictionary."""
        config = super().get_config()
        config["target"] = self.target

        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config. Returning the instance."""
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])
        schema = config.pop("schema")
        target = config.pop("target")
        return cls(schema, target, **config)
