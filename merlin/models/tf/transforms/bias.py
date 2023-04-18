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
from typing import Sequence, Union

import tensorflow as tf

from merlin.models.config.schema import requires_schema
from merlin.models.tf.core.base import Block, PredictionOutput
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.schema_utils import schema_to_tensorflow_metadata_json
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class LogitsTemperatureScaler(Block):
    """Scale the logits higher or lower,
    this is often used to reduce the overconfidence of the model.

    Parameters
    ----------
    temperature : float
        Divide the logits by this scaler.
    """

    def __init__(self, temperature: float, **kwargs):
        super(LogitsTemperatureScaler, self).__init__(**kwargs)
        self.temperature = temperature

    def call(
        self, outputs: Union[Prediction, PredictionOutput], training=False, testing=False, **kwargs
    ) -> Union[tf.Tensor, Prediction]:
        if (training or testing) and isinstance(outputs, Prediction):
            predictions = self.apply_temperature(outputs.predictions)
            return outputs.copy_with_updates(outputs=predictions)
        else:
            return outputs

    def call_outputs(
        self, outputs: PredictionOutput, training=False, testing=False, **kwargs
    ) -> "PredictionOutput":
        if (training or testing) and isinstance(outputs, Prediction):
            predictions = self.apply_temperature(outputs.predictions)
            return outputs.copy_with_updates(predictions=predictions)
        else:
            return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def apply_temperature(self, predictions):
        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
        predictions = predictions / self.temperature
        return predictions

    def get_config(self):
        config = super().get_config()
        config["temperature"] = self.temperature
        return config


@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class PopularityLogitsCorrection(Block):
    """Correct the predicted logit scores based on the item frequency,
    using the logQ correction proposed in sampled softmax [1]_ [2]_.
    The correction is done as `logits -= log(item_prob)`,
    where `item_prob = item_freq_count / sum(item_freq_count)` is
    a probability distribution of the item frequency. In a nutshell,
    the logQ correction aims to increase the prediction scores (logits)
    for infrequent items and decrease the ones for frequent items, so
    that they are not much more penalized for being sampled more often.

    References
    ----------
    .. [1] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic
       Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial
       Intelligence and Statistics (AISTATS).

    .. [2] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312

    Parameters:
    ----------
    item_freq_probs : Union[tf.Tensor, Sequence]
        A Tensor or list with item frequencies (if is_prob_distribution=False)
        or with item probabilities (if is_prob_distribution=True)
    is_prob_distribution: bool, optional
        If True, the item_freq_probs should be a probability distribution of the items.
        If False, the item frequencies is converted to probabilities
    reg_factor: float
        Factor to scale the logq correction, by default 1.0
    schema: Schema, optional
        The `Schema` with input features,
        by default None
    candidate_tag_id: Tag, optional
        The tag of the candidate id
    """

    def __init__(
        self,
        item_freq_probs: Union[tf.Tensor, Sequence] = None,
        is_prob_distribution: bool = False,
        reg_factor: float = 1.0,
        schema: Schema = None,
        candidate_tag_id=Tags.ITEM_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)

        self.reg_factor = reg_factor
        self.candidate_tag_id = candidate_tag_id
        self.candidate_id_name = self.schema.select_by_tag(self.candidate_tag_id).first.name
        self.item_freq_probs = item_freq_probs

        if self.item_freq_probs is not None:
            self._check_items_cardinality(item_freq_probs)
            candidate_probs = tf_utils.get_candidate_probs(item_freq_probs, is_prob_distribution)

            self.candidate_probs = tf.Variable(
                candidate_probs,
                name="candidate_probs",
                trainable=False,
                dtype=tf.float32,
                validate_shape=False,
                shape=tf.shape(candidate_probs),
            )

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        frequencies_probs_col: str,
        is_prob_distribution: bool = False,
        gpu: bool = True,
        schema: Schema = None,
        **kwargs,
    ):
        """Load the item frequency table from a parquet file
        (in the format automatically generated by NVTabular with workflow.fit()).
        It supposed the parquet file has a single column with the item frequencies
        and is indexed by item ids.

        Parameters
        ----------
        parquet_path : str
            Path to the parquet file
        frequencies_probs_col : str
            Column name containing the items frequencies / probabilities
        is_prob_distribution: bool, optional
            If True, the frequencies_probs_col should contain the probability
            distribution of the items. If False, the frequencies_probs_col values
            are frequencies and will be converted to probabilities
        gpu : bool, optional
            Whether to load data using cudf, by default True
        schema: Schema, optional
            The `Schema` with input features,
            by default None

        Returns
        -------
            An instance of PopularityLogitsCorrection
        """
        # TODO: Use the schema to infer the path to the item frequency parquet table
        if gpu:
            import cudf

            df = cudf.read_parquet(parquet_path)
            item_frequency = tf.squeeze(tf_utils.df_to_tensor(df[frequencies_probs_col]))
        else:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            item_frequency = tf.squeeze(tf.convert_to_tensor(df[frequencies_probs_col].values))
        return cls(
            item_freq_probs=item_frequency,
            is_prob_distribution=is_prob_distribution,
            schema=schema,
            **kwargs,
        )

    def get_candidate_probs(self):
        return self.candidate_probs.value()

    def update(
        self, item_freq_probs: Union[tf.Tensor, Sequence], is_prob_distribution: bool = False
    ):
        """Updates the item frequencies / probabilities

        Parameters:
        ----------
        item_freq_probs : Union[tf.Tensor, Sequence]
            A Tensor or list with item frequencies (if is_prob_distribution=False)
            or with item probabilities (if is_prob_distribution=True)
        is_prob_distribution: bool, optional
            If True, the item_freq_probs should be a probability distribution of the items.
            If False, the item frequencies is converted to probabilities
        """
        self._check_items_cardinality(item_freq_probs)
        candidate_probs = tf_utils.get_candidate_probs(item_freq_probs, is_prob_distribution)
        self.item_freq_probs = item_freq_probs
        self.candidate_probs.assign(candidate_probs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(
        self, outputs: Union[Prediction, PredictionOutput], features, training=False, **kwargs
    ):
        if training and isinstance(outputs, Prediction):
            # this logic is for the new ModelOutput api
            positive_candidate_ids = tf.squeeze(features[self.candidate_id_name])
            negative_candidate_ids = outputs.negative_candidate_ids

            predictions = self.compute_log_q_correction(
                outputs.predictions, positive_candidate_ids, negative_candidate_ids
            )

            return outputs.copy_with_updates(outputs=predictions)

        return outputs

    def compute_log_q_correction(
        self, predictions, positive_candidate_ids, negative_candidate_ids=None
    ):
        positive_probs = tf.gather(self.candidate_probs, positive_candidate_ids)
        if negative_candidate_ids is not None:
            negative_probs = tf.gather(self.candidate_probs, negative_candidate_ids)
            # repeat negative scores for each positive item
            negative_probs = tf.reshape(
                tf.tile(tf.squeeze(negative_probs), tf.shape(positive_candidate_ids)[0:1]),
                (-1, tf.shape(negative_candidate_ids)[0]),
            )
            positive_probs = tf.concat([tf.expand_dims(positive_probs, -1), negative_probs], axis=1)

        # Applies the logQ correction
        epsilon = 1e-16
        predictions = predictions - (self.reg_factor * tf.math.log(positive_probs + epsilon))
        return predictions

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        predictions = outputs.predictions
        if training:
            positive_item_ids, negative_item_ids = (
                outputs.positive_item_ids,
                outputs.negative_item_ids,
            )

            predictions = self.compute_log_q_correction(
                outputs.predictions, positive_item_ids, negative_item_ids
            )
        return outputs.copy_with_updates(predictions=predictions)

    def _check_items_cardinality(self, item_freq_probs):
        cardinalities = schema_utils.categorical_cardinalities(self.schema)
        item_id_feature_name = self.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
        if tf.shape(item_freq_probs)[0] != cardinalities[item_id_feature_name]:
            raise ValueError(
                "The item frequency table length does not match the item ids cardinality"
                f"(expected {cardinalities[item_id_feature_name]}"
                f", got {tf.shape(item_freq_probs)[0]})"
            )

    def compute_call_output_shape(self, input_shapes):
        return self.compute_output_shape(input_shapes)

    def get_config(self):
        config = super().get_config()
        if self.schema:
            config["schema"] = schema_to_tensorflow_metadata_json(self.schema)
        config["reg_factor"] = self.reg_factor
        config["candidate_tag_id"] = self.candidate_tag_id.value
        config["item_freq_probs"] = self.item_freq_probs.numpy()

        return config

    @classmethod
    def from_config(cls, config):
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        config["candidate_tag_id"] = Tags(config["candidate_tag_id"])
        config["item_freq_probs"] = tf.convert_to_tensor(config["item_freq_probs"])

        return cls(**config)
