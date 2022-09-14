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
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class LogitsTemperatureScaler(Block):
    """Scale the logits higher or lower,
    this is often used to reduce the overconfidence of the model.

    Parameters
    ----------
    temperature : float
        Divide the logits by this scaler.
    apply_on_call_outputs: bool
        Whether to apply the transform (logits / temperature) on
        `call()` or `call_outputs()`. By default True
    """

    def __init__(self, temperature: float, apply_on_call_outputs: bool = True, **kwargs):
        super(LogitsTemperatureScaler, self).__init__(**kwargs)
        self.temperature = temperature
        self.apply_on_call_outputs = apply_on_call_outputs

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        if not self.apply_on_call_outputs:
            return self.apply_temperature(inputs)
        else:
            return inputs

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        predictions = self.apply_temperature(predictions)

        return outputs.copy_with_updates(predictions=predictions, targets=targets)

    def compute_output_shape(self, input_shape):
        return input_shape

    def apply_temperature(self, predictions):
        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"
        predictions = predictions / self.temperature
        return predictions


@tf.keras.utils.register_keras_serializable(package="merlin_models")
@requires_schema
class PopularityLogitsCorrection(Block):
    """Correct the predicted logit scores based on the item frequency,
    using the logQ correction proposed in sampled softmax [1]_ [2]_.
    The correction is done as `logits -= log(item_prob)`,
    where `item_prob = item_freq_count / sum(item_freq_count)` is
    a probability distribution of the item frequency. In a nutshell,
    the logQ correction aims to increase the prediction scores (logits)
    for infrequent items and decrease the ones for frequent items.

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
    """

    def __init__(
        self,
        item_freq_probs: Union[tf.Tensor, Sequence] = None,
        is_prob_distribution: bool = False,
        reg_factor: float = 1.0,
        schema: Schema = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema:
            self.set_schema(schema)

        self.reg_factor = reg_factor

        if item_freq_probs is not None:
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
        self.candidate_probs.assign(candidate_probs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        predictions = outputs.predictions
        if training:
            positive_item_ids, negative_item_ids = (
                outputs.positive_item_ids,
                outputs.negative_item_ids,
            )
            positive_probs = tf.gather(self.candidate_probs, positive_item_ids)

            if negative_item_ids is not None:
                negative_probs = tf.gather(self.candidate_probs, negative_item_ids)
                # repeat negative scores for each positive item
                negative_probs = tf.reshape(
                    tf.tile(negative_probs, tf.shape(positive_item_ids)[0:1]),
                    (-1, tf.shape(negative_item_ids)[0]),
                )
                positive_probs = tf.concat(
                    [tf.expand_dims(positive_probs, -1), negative_probs], axis=1
                )

            # Applies the logQ correction
            epsilon = 1e-16
            predictions = predictions - (self.reg_factor * tf.math.log(positive_probs + epsilon))

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
