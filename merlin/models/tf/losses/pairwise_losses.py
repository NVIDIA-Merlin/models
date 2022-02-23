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

from .loss_base import LossRegistryMixin


class PairwiseLoss(Loss, LossRegistryMixin):
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


@LossRegistryMixin.registry.register("bpr")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BPRLoss(PairwiseLoss):
    """The Bayesian Personalised Ranking (BPR) pairwise loss [1]_

    References
    ----------
    .. [1] Rendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-Thieme, L. BPR: Bayesian
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


@LossRegistryMixin.registry.register("bpr-max")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class BPRmaxLoss(PairwiseLoss):
    """The BPR-max pairwise loss proposed in [1]_

    References
    ----------
    .. [1] Hidasi, Balázs, and Alexandros Karatzoglou. "Recurrent neural networks with top-k gains
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
        neg_softmax_weights = tf.nn.softmax(negatives_scores, axis=-1)
        reg = tf.square(negatives_scores) * neg_softmax_weights * self.reg_lambda
        loss = -tf.math.log(tf.nn.sigmoid(sub) * neg_softmax_weights) + reg
        return loss


@LossRegistryMixin.registry.register("top1")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TOP1Loss(PairwiseLoss):
    """The TOP pairwise loss proposed in [1]_

    References
    ----------
    .. [1] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Session-based recommendations
       with recurrent neural networks,” in Proceedings of Fourth International Conference on
       Learning Representations (ICLR’16), 2016. https://arxiv.org/abs/1511.06939
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
        sub = negatives_scores - positives_scores
        loss = tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores))
        return loss


@LossRegistryMixin.registry.register("top1_v2")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TOP1v2Loss(PairwiseLoss):
    """An adapted version of the TOP pairwise loss proposed in [1]_, but following the
    current GRU4Rec implementation [2]_.

    References:
    .. [1] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Session-based recommendations
        with recurrent neural networks,” in Proceedings of Fourth International Conference on
        Learning Representations (ICLR’16), 2016.
    .. [2] https://arxiv.org/abs/1511.06939
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
        sub = negatives_scores - positives_scores
        loss = (
            tf.reduce_mean(
                tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores)),
                keepdims=True,
                axis=-1,
            )
            - tf.nn.sigmoid(tf.square(positives_scores))
            / tf.cast(tf.shape(negatives_scores)[1], tf.float32)
        )
        return loss


@LossRegistryMixin.registry.register("top1-max")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TOP1maxLoss(PairwiseLoss):
    """The TOP1-max pairwise loss proposed in [1]_

    References
    ----------
    .. [1] Hidasi, Balázs, and Alexandros Karatzoglou. "Recurrent neural networks with top-k gains
       for session-based recommendations." Proceedings of the 27th ACM international conference on
       information and knowledge management. 2018. https://arxiv.org/abs/1706.03847
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
        sub = negatives_scores - positives_scores
        neg_softmax_weights = tf.nn.softmax(negatives_scores, axis=-1)
        loss = (
            tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores))
        ) * neg_softmax_weights
        return loss


@LossRegistryMixin.registry.register("logistic")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class LogisticLoss(PairwiseLoss):
    """Pairwise log loss, as described in [1]_: `log(1 + exp(r_uj - r_ui))`, where r_ui is the score
    of the positive item and r_uj the score of negative items.

    References
    ----------
    .. [1] Sun, Zhu, et al. "Are we evaluating rigorously? benchmarking recommendation for
       reproducible evaluation and fair comparison." Fourteenth ACM conference on recommender
       systems. 2020.
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
        sub = negatives_scores - positives_scores
        # Equivalent to log(1 + exp(sub))
        loss = tf.nn.relu(sub) + tf.math.log1p(tf.math.exp(-tf.abs(sub)))
        return loss


@LossRegistryMixin.registry.register("hinge")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class HingeLoss(PairwiseLoss):
    """Pairwise hinge loss, as described in [1]_: `max(0, 1 + r_uj - r_ui))`, where r_ui is the score
    of the positive item and r_uj the score of negative items.

    References
    ----------
    .. [1] Sun, Zhu, et al. "Are we evaluating rigorously? benchmarking recommendation for
       reproducible evaluation and fair comparison." Fourteenth ACM conference on recommender
       systems. 2020.
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
        loss = tf.nn.relu(1 + negatives_scores - positives_scores)
        return loss


@LossRegistryMixin.registry.register("adaptive_hinge")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AdaptiveHingeLoss(PairwiseLoss):
    """Adaptive hinge pairwise loss. Samples the highest
    negative scores, as they are closer to violating the
    expected ranking  where positives should be ranked higher.

    Approximates the idea of Weighted Approximate-Rank Pairwise (WARP) loss [1],
    inspired by
    `Spotlight https://maciejkula.github.io/spotlight/losses.html#spotlight.losses.
    adaptive_hinge_loss`_ implementation.

    References
    ----------
    .. [1] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011
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
        max_neg_scores = tf.reduce_max(negatives_scores, axis=-1, keepdims=True)
        loss = tf.nn.relu(1 + max_neg_scores - positives_scores)
        return loss
