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

import tensorflow as tf

from merlin.models.tf.core import Block, PredictionOutput
from merlin.models.tf.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemsPredictionTopK(Block):
    """
    Block to extract top-k scores from the item-prediction layer output
    and corresponding labels.

    Parameters
    ----------
    k: int
        Number of top candidates to return.
        Defaults to 20
    """

    def __init__(
        self,
        k: int = 20,
        **kwargs,
    ):
        super(ItemsPredictionTopK, self).__init__(**kwargs)
        self._k = k

    @tf.function
    def call_outputs(
        self, outputs: PredictionOutput, training=False, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions

        tf.assert_equal(
            tf.shape(targets),
            tf.shape(predictions),
            f"Predictions ({tf.shape(predictions)}) and targets ({tf.shape(targets)}) "
            f"should have the same shape. Check if targets were one-hot encoded "
            f"(with LabelToOneHot() block for example).",
        )

        topk_scores, topk_labels, label_relevant_counts = tf_utils.extract_topk(
            self._k, predictions, targets
        )
        return PredictionOutput(topk_scores, topk_labels, label_relevant_counts)
