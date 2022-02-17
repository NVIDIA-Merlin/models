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
from typing import Tuple

import tensorflow as tf

from merlin_models.tf.core import Block

from ..utils import tf_utils


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
    transform_to_onehot: bool
        If set to True, transform integer encoded ids to one-hot representation.
        Defaults to True
    """

    def __init__(
        self,
        k: int = 20,
        transform_to_onehot: bool = True,
        **kwargs,
    ):
        super(ItemsPredictionTopK, self).__init__(**kwargs)
        self._k = k
        self.transform_to_onehot = transform_to_onehot

    @tf.function
    def call_targets(self, predictions, targets, training=False, **kwargs) -> tf.Tensor:
        if self.transform_to_onehot:
            num_classes = tf.shape(predictions)[-1]
            targets = tf_utils.tranform_label_to_onehot(targets, num_classes)

        topk_scores, _, topk_labels = tf_utils.extract_topk(self._k, predictions, targets)
        return topk_labels, topk_scores


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class BruteForceTopK(Block):
    """
    Block to retrieve top-k negative candidates for Item Retrieval evaluation.

    Parameters
    ----------
    k: int
        Number of top candidates to retrieve.
        Defaults to 20

    """

    def __init__(
        self,
        k: int = 20,
        **kwargs,
    ):
        super(BruteForceTopK, self).__init__(**kwargs)
        self._k = k
        self._candidates = None

    def load_index(self, candidates: tf.Tensor, identifiers: tf.Tensor = None):
        """
        Set the embeddings and identifiers variables
        """
        if len(tf.shape(candidates)) != 2:
            raise ValueError(
                f"The candidates embeddings tensor must be 2D (got {candidates.shape})."
            )
        if not identifiers:
            identifiers = tf.range(candidates.shape[0])

        self._identifiers = self.add_weight(
            name="identifiers",
            dtype=identifiers.dtype,
            shape=identifiers.shape,
            initializer=tf.keras.initializers.Constant(value=0),
            trainable=False,
        )
        self._candidates = self.add_weight(
            name="candidates",
            dtype=candidates.dtype,
            shape=candidates.shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )

        self._identifiers.assign(identifiers)
        self._candidates.assign(candidates)
        return self

    def load_from_dataset(
        self,
        items_embeddings,
    ):
        raise NotImplementedError()

    def _compute_score(self, queries: tf.Tensor, candidates: tf.Tensor) -> tf.Tensor:
        """Computes the standard dot product score from queries and candidates."""
        return tf.matmul(queries, candidates, transpose_b=True)

    def call(
        self, inputs: tf.Tensor, training: bool = False, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute top-k scores and related indices from query inputs
        """
        if self._candidates is None:
            raise ValueError("load_index should be called before")
        scores = self._compute_score(inputs, self._candidates)
        top_scores, top_indices = tf.math.top_k(scores, k=self._k)
        return top_scores, tf.gather(self._identifiers, top_indices)

    def call_targets(self, predictions, targets, training=False, **kwargs) -> tf.Tensor:
        queries = self.context["query"]
        top_scores, _ = self.call(queries)
        predictions = tf.expand_dims(predictions[:, 0], -1)
        predictions = tf.concat([predictions, top_scores], axis=-1)
        # Positives in the first column and negatives in the subsequent columns
        targets = tf.concat(
            [
                tf.ones([tf.shape(predictions)[0], 1]),
                tf.zeros([tf.shape(predictions)[0], self._k]),
            ],
            axis=1,
        )
        return targets, predictions
