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

import tensorflow as tf

from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils.tf_utils import list_col_to_ragged
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class UniformNegativeSampling(tf.keras.layers.Layer):
    """Random in-batch negative sampling.

    Only works with positive-only binary-target batches.
    """

    def __init__(self, schema: Schema, n_per_positive: int, seed: Optional[int] = None, **kwargs):
        """Instantiate a sampling block."""
        super(UniformNegativeSampling, self).__init__(**kwargs)
        self.n_per_positive = n_per_positive
        self.item_id_col = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        self.schema = schema.select_by_tag(Tags.ITEM)
        self.seed = seed

    def call(self, inputs: TabularData, targets=None) -> Prediction:
        """Extend batch of inputs and targets with negatives."""
        # 1. Select item-features
        fist_input = list(inputs.values())[0]
        batch_size = (
            fist_input.shape[0] if not isinstance(fist_input, tuple) else fist_input[1].shape[0]
        )

        if batch_size is None:
            return Prediction(inputs, targets)

        sampled_num_negatives = self.n_per_positive * batch_size
        # 2. Sample `n_per_positive * batch_size` items at random
        sampled_positive_idx = tf.random.uniform(
            (sampled_num_negatives,),
            maxval=batch_size,
            dtype=tf.int32,
            seed=self.seed,
        )

        # mask to remove negatives that conflict with positives
        mask = tf.logical_not(
            tf.equal(
                tf.repeat(inputs[self.item_id_col], self.n_per_positive, axis=0),
                tf.gather(inputs[self.item_id_col], sampled_positive_idx),
            )
        )
        if mask.shape.ndims > 1:
            mask = tf.reduce_all(mask, 1)  # get the batch dimension
        # keep all the positive inputs
        mask = tf.concat([tf.repeat(True, batch_size), mask], 0)

        # 3. Loop through all features:
        #   - For item-feature: append from item-collection
        #   - For user-feature: repeat `n_per_positive` times
        item_cols = self.schema.column_names
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                val = list_col_to_ragged(val)

            if name in item_cols:
                negatives = tf.gather(val, sampled_positive_idx)
            else:
                if isinstance(val, tf.RaggedTensor):
                    negatives = tf.concat([val] * self.n_per_positive, axis=0)
                else:
                    negatives = tf.repeat(val, self.n_per_positive, axis=0)

            outputs[name] = tf.concat([val, negatives], axis=0)
            outputs[name] = tf.ragged.boolean_mask(outputs[name], mask)

        # update targets if present
        if targets is not None:

            def mask_targets(target_tensor):
                out = tf.concat(
                    [
                        target_tensor,
                        tf.zeros((sampled_num_negatives, 1), dtype=target_tensor.dtype),
                    ],
                    0,
                )
                out = tf.boolean_mask(out, mask)

                return out

            if isinstance(targets, dict):
                targets = {k: mask_targets(v) for k, v in targets.items()}
            elif isinstance(targets, list):
                targets = [mask_targets(v) for v in targets]
            elif isinstance(targets, tuple):
                targets = tuple([mask_targets(v) for v in targets])
            elif isinstance(targets, tf.Tensor):
                targets = mask_targets(targets)
            else:
                raise ValueError("Unsupported target type: {}".format(type(targets)))

        return Prediction(outputs, targets)

    def get_config(self):
        """Returns the config of the layer as a Python dictionary."""
        config = super().get_config()
        config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        config["n_per_positive"] = self.n_per_positive
        config["seed"] = self.seed
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config. Returning the instance."""
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        n_per_positive = config.pop("n_per_positive")
        seed = None
        if "seed" in config:
            seed = config.pop("seed")
        kwargs = config
        return cls(schema, n_per_positive, seed=seed, **kwargs)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        return input_shape
