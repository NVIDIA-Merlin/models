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
from typing import Union

import tensorflow as tf

from merlin.models.config.schema import requires_schema
from merlin.models.tf.core.base import Block
from merlin.models.tf.core.prediction import Prediction
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class ContrastiveSampleWeight(Block):
    def __init__(
        self,
        pos_class_weight: Union[str, tf.Tensor, float],
        neg_class_weight: Union[tf.Tensor, float] = 1,
        schema: Schema = None,
        candidate_tag_id=Tags.ITEM_ID,
        **kwargs,
    ):
        """Generate the sample weights for positive and sampled
        negative candidates.
        This class can be used as a `post` of the `ContrastiveOutput`
        instance.

        Parameters
        ----------
        pos_class_weight: Union[str, tf.Tensor, float]
            The positive candidates weights. This class supports three types of weights:
            * str: the column name of the input feature to use as the sample weight.
            In this case, the weights are dynamic and change based on the input
            (user-candidate) interaction.
            * tf.Tensor: a weight vector of shape (#unique-candidates,). In this case,
            the weights are specific to each candidate and are the same for all
            users' interactions with the given candidate.
            * float: use the same weight value for all positives candidates. In this case,
            the weights are static for all (user-candidate) interactions.
        neg_class_weight:  Union[tf.Tensor, float]
            The negative candidates weights. The class supports two types of weights:
            * tf.Tensor: a weight vector of shape (#unique-candidates,). In this case,
            the weights are specific to each sampled negative candidate.
            * float: Use the same weight value for all negative candidates.
        schema: Schema, optional
            The `Schema` with candidate id feature,
            by default None
        candidate_tag_id: Tag, optional
            The tag of the candidate id column. This feature is needed to look-up for the
            positive candidates weights when `pos_class_weight` is a `tf.Tensor` object.
            by default 'item_id'

        Example usage::
            outputs = mm.ContrastiveOutput(
                        DotProduct(),
                        schema=data.schema.select_by_tag(Tags.ITEM_ID),
                        negative_samplers="in-batch",
                        store_negative_ids=True,
                        post=ContrastiveSampleWeight(
                            pos_class_weight='interaction-weight',
                            neg_class_weight=0.5,
                            schema=data.schema,
                        ),
                    )
        """
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)
        self.pos_class_weight = pos_class_weight
        self.neg_class_weight = neg_class_weight
        self.candidate_tag_id = candidate_tag_id
        self.candidate_id_name = self.schema.select_by_tag(self.candidate_tag_id).first.name

    def call(
        self,
        outputs: Prediction,
        features,
        training=False,
        testing=False,
    ) -> Prediction:
        """Update the predictions returned by the
        ConstrastiveOutput with a 2-D sample weights
        for negative and positive candidates.
        """
        if training or testing:
            predictions = outputs.predictions
            shapes = tf.shape(predictions)

            if isinstance(self.pos_class_weight, str):
                pos_samples = features.get(self.pos_class_weight)
                if pos_samples is None:
                    raise ValueError(
                        "The model's inputs don't contain the positive weight"
                        f" feature {self.pos_class_weight}."
                    )
            elif isinstance(self.pos_class_weight, tf.Tensor):
                positive_ids = features[self.candidate_id_name]
                pos_samples = tf.gather(self.pos_class_weight, positive_ids)
            else:
                pos_samples = tf.ones((shapes[0], 1)) * self.pos_class_weight

            if isinstance(self.neg_class_weight, tf.Tensor):
                negative_candidate_ids = outputs.negative_candidate_ids
                neg_samples = tf.gather(self.neg_class_weight, negative_candidate_ids)
                if tf.shape(neg_samples)[-1] == 1:
                    # repeat negative samples for each positive candidate
                    neg_samples = tf.tile(
                        tf.expand_dims(tf.squeeze(neg_samples), 0), [tf.shape(neg_samples)[0], 1]
                    )

            else:
                neg_samples = tf.ones((shapes[0], shapes[1] - 1)) * self.neg_class_weight

            # generate a 2-d matrix of sample weights
            pos_samples = tf.cast(pos_samples, tf.float32)
            samples_weights = tf.concat([pos_samples, neg_samples], axis=1)
            # expand tensors dimension to ignore the default mean over axis -1,
            # applied by keras losses.
            outputs = outputs.copy_with_updates(
                outputs=tf.expand_dims(outputs.outputs, -1),
                targets=tf.expand_dims(outputs.targets, -1),
                sample_weight=tf.cast(samples_weights, tf.float32),
            )

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        """
        Save the class configuration when `model.save()` is called.
        """
        config = super().get_config()
        if self.schema:
            config["schema"] = schema_utils.schema_to_tensorflow_metadata_json(self.schema)
        config["candidate_tag_id"] = self.candidate_tag_id.value

        if isinstance(self.pos_class_weight, tf.Tensor):
            config["pos_class_weight"] = self.pos_class_weight.numpy()
        else:
            config["pos_class_weight"] = self.pos_class_weight

        if isinstance(self.neg_class_weight, tf.Tensor):
            config["neg_class_weight"] = self.neg_class_weight.numpy()
        else:
            config["neg_class_weight"] = self.neg_class_weight

        return config

    @classmethod
    def from_config(cls, config):
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        config["candidate_tag_id"] = Tags(config["candidate_tag_id"])
        if not isinstance(config["pos_class_weight"], (str, float, int)):
            config["pos_class_weight"] = tf.convert_to_tensor(config["pos_class_weight"])
        if not isinstance(config["neg_class_weight"], (float, int)):
            config["neg_class_weight"] = tf.convert_to_tensor(config["neg_class_weight"])

        return cls(**config)
