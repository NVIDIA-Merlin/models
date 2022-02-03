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
from tensorflow.keras.losses import Loss


class PairwiseLoss(Loss):
    """Base class for pairwise losses"""

    def _check_only_one_positive_label_per_example(self, y_true: tf.Tensor):
        """Checks if there is only one positive label per example

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        """
        tf.assert_equal(
            tf.reduce_sum(y_true, axis=1),
            tf.ones_like(y_true[:, 0]),
            message="Only one positive label is allowed per example",
        )

    def _get_positives_negatives_scores(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Extracts the positive item score (only one) and the negative item scores

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Returns a tuple (positives_scores, negatives_scores)
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        self._check_only_one_positive_label_per_example(y_true)

        positives_mask = tf.cast(y_true, tf.bool)
        positives_scores = tf.expand_dims(tf.boolean_mask(y_pred, positives_mask), -1)
        negatives_scores = tf.reshape(
            tf.boolean_mask(y_pred, tf.logical_not(positives_mask)),
            (tf.shape(y_pred)[0], tf.shape(y_pred)[1] - 1),
        )

        return positives_scores, negatives_scores


class BPR(PairwiseLoss):
    """The BPR pairwise loss [1]

    References:
        [1] Rendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-Thieme, L. BPR: Bayesian
        personalized ranking from implicit feedback. In UAI’09: 25th Conf. on Uncertainty in
        Artificial Intelligence. https://arxiv.org/pdf/1205.2618.pdf
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Loss computation

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor (batch size x 1)
            Loss per example
        """
        positives_scores, negatives_scores = self._get_positives_negatives_scores(y_true, y_pred)
        sub = positives_scores - negatives_scores
        loss = -tf.math.log(tf.nn.sigmoid(sub))
        return loss


class BPRmax(PairwiseLoss):
    """The BPR-max pairwise loss proposed in [1]

    References:
        [1] Hidasi, Balázs, and Alexandros Karatzoglou. "Recurrent neural networks with top-k gains
        for session-based recommendations." Proceedings of the 27th ACM international conference on
        information and knowledge management. 2018. https://arxiv.org/abs/1706.03847
    """

    def __init__(self, reg_lambda: float = 1.0, **kwargs):
        """[summary]

        Parameters
        ----------
        reg_lambda : float, optional
            Regularization factor of the `softmax(neg_scores) term`, by default 1.0
        """
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Loss computation

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor (batch size x 1)
            Loss per example
        """
        positives_scores, negatives_scores = self._get_positives_negatives_scores(y_true, y_pred)
        sub = positives_scores - negatives_scores
        neg_softmax_weights = tf.nn.softmax(negatives_scores, axis=-1)
        reg = tf.square(negatives_scores) * neg_softmax_weights * self.reg_lambda
        loss = -tf.math.log(tf.nn.sigmoid(sub) * neg_softmax_weights) + reg
        return loss


class TOP1(PairwiseLoss):
    """The TOP pairwise loss proposed in [1]

    References:
        [1] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Session-based recommendations
        with recurrent neural networks,” in Proceedings of Fourth International Conference on
        Learning Representations (ICLR’16), 2016. https://arxiv.org/abs/1511.06939
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Loss computation

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor (batch size x 1)
            Loss per example
        """
        positives_scores, negatives_scores = self._get_positives_negatives_scores(y_true, y_pred)
        sub = negatives_scores - positives_scores
        loss = tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores))
        return loss


class TOP1v2(PairwiseLoss):
    """An adapted version of the TOP pairwise loss proposed in [1], but following the the
    current GRU4Rec implementation [1].

    References:
        [1] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Session-based recommendations
        with recurrent neural networks,” in Proceedings of Fourth International Conference on
        Learning Representations (ICLR’16), 2016. https://arxiv.org/abs/1511.06939
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Loss computation

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor (batch size x 1)
            Loss per example
        """
        positives_scores, negatives_scores = self._get_positives_negatives_scores(y_true, y_pred)
        sub = negatives_scores - positives_scores
        loss = (
            tf.reduce_mean(
                tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores)),
                keepdims=True,
                axis=1,
            )
            - tf.nn.sigmoid(tf.square(positives_scores))
            / tf.cast(tf.shape(negatives_scores)[1], tf.float32)
        )
        return loss


class TOP1max(PairwiseLoss):
    """The TOP1-max pairwise loss proposed in [1]

    References:
        [1] Hidasi, Balázs, and Alexandros Karatzoglou. "Recurrent neural networks with top-k gains
        for session-based recommendations." Proceedings of the 27th ACM international conference on
        information and knowledge management. 2018. https://arxiv.org/abs/1706.03847
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Loss computation

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor (batch size x 1)
            Loss per example
        """
        positives_scores, negatives_scores = self._get_positives_negatives_scores(y_true, y_pred)
        sub = negatives_scores - positives_scores
        neg_softmax_weights = tf.nn.softmax(negatives_scores, axis=-1)
        loss = (
            tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores))
        ) * neg_softmax_weights
        return loss
