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

import abc
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.losses import Loss

from merlin.models.tf.losses.base import LossRegistryMixin
from merlin.models.tf.utils.tf_utils import add_epsilon_to_zeros
from merlin.models.utils.constants import MAX_FLOAT, MIN_FLOAT
from merlin.models.utils.doc_utils import docstring_parameter

PAIRWISE_LOSSES_COMPUTE_DOCSTRING = """Computes the loss

        Parameters
        ----------
        positives_scores : tf.Tensor
            Prediction scores for the positive items (batch size x 1)
        negatives_scores : tf.Tensor
            Prediction scores for the positive items (batch size x number negative samples)

        Returns
        -------
        tf.Tensor (batch size x number negative samples)
            Loss per negative sample
        """


class PairwiseLoss(Loss, LossRegistryMixin):
    """Base class for pairwise losses"""

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor = None,
        valid_negatives_mask: Optional[tf.Tensor] = None,
    ):
        self.valid_negatives_mask = valid_negatives_mask
        # This will call the `call` method implemented by the super class.
        loss = super().__call__(y_true, y_pred, sample_weight)
        return loss

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Loss computation. In the end, it valid_negatives_mask is provided, it masks the loss
        ensuring that false negative have zeroed loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        tf.Tensor
            Loss per example
        """
        (
            positives_scores,
            negatives_scores,
            valid_rows_with_positive_mask,
        ) = self._separate_positives_negatives_scores(y_true, y_pred)
        loss = self.compute(positives_scores, negatives_scores)
        loss = self._mask_loss(loss, valid_rows_with_positive_mask)
        return loss

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    @abc.abstractmethod
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        raise NotImplementedError()

    @property
    def valid_negatives_mask(self) -> Optional[tf.Tensor]:
        return self._valid_negatives_mask

    @valid_negatives_mask.setter
    def valid_negatives_mask(self, value: Optional[tf.Tensor]):
        """Sets the valid_negatives_mask so that the loss can be
        zeroed for false negatives (negative item equal to the positive item)

        Parameters
        ----------
        value : tf.Tensor, optional
            2D Boolean mask tensor matching the dims of `y_pred`, which is False only for positions
            where the the negative item id is equal to the positive item id, by default None
        """
        self._valid_negatives_mask = value

    def _check_max_one_positive_label_per_example(self, y_true: tf.Tensor):
        """Checks if there is at most one positive label per example

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        """
        tf.debugging.assert_less_equal(
            tf.reduce_max(tf.reduce_sum(y_true, axis=1)),
            1.0,
            message="The batch contains more examples with more than one positive item,"
            " which is not supported by the pairwise losses.",
        )

    def _separate_positives_negatives_scores(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Extracts the positive item score (only one) and the negative item scores

        Parameters
        ----------
        y_true : tf.Tensor
            Prediction labels. Expects a 2D tensor of shape (batch size, num predicted scores)
        y_pred : tf.Tensor
            Prediction scores. Expects a 2D tensor of shape (batch size, num predicted scores)

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            Returns a tuple (positives_scores, negatives_scores, valid_rows_with_positive_mask)
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        self._check_max_one_positive_label_per_example(y_true)

        # Mask that checks if there is a positive available for the example.
        # During training that is ensured when using `ItemRetrievalScorer` but during
        # evaluation only the top-k retrieved items are provided, and chances are
        # that the positive item is not among the top-k, resulting in a row with only
        # negatvies
        valid_rows_with_positive_mask = tf.cast(tf.reduce_sum(y_true, axis=1), tf.bool)
        y_pred_valid_rows = tf.boolean_mask(y_pred, valid_rows_with_positive_mask)
        y_true_valid_rows = tf.boolean_mask(y_true, valid_rows_with_positive_mask)

        # Extracting the positive (only one) and the negatives scores in separate tensors
        positives_mask = tf.cast(y_true_valid_rows, tf.bool)
        positives_scores = tf.expand_dims(tf.boolean_mask(y_pred_valid_rows, positives_mask), -1)
        negatives_scores = tf.reshape(
            tf.boolean_mask(y_pred_valid_rows, tf.logical_not(positives_mask)),
            (tf.shape(y_pred_valid_rows)[0], tf.shape(y_pred_valid_rows)[1] - 1),
        )

        # Initializing the positive and negative scores with very large and small values
        # respectively, so that for examples that does not include the positive (e.g. during eval)
        # the difference between the large positive and small negative scores lead to 0 loss
        # for this example
        positive_large_scores = tf.fill(
            value=tf.constant(MAX_FLOAT, dtype=y_pred.dtype), dims=(tf.shape(y_pred)[0], 1)
        )
        negatives_small_scores = tf.fill(
            value=tf.constant(MIN_FLOAT, dtype=y_pred.dtype),
            dims=(tf.shape(y_pred)[0], tf.shape(y_pred)[1] - 1),
        )

        update_indices = tf.expand_dims(
            tf.boolean_mask(tf.range(tf.shape(y_true)[0]), valid_rows_with_positive_mask), -1
        )
        positives_scores_final = tf.tensor_scatter_nd_update(
            positive_large_scores, update_indices, positives_scores
        )
        negatives_scores_final = tf.tensor_scatter_nd_update(
            negatives_small_scores, update_indices, negatives_scores
        )

        return positives_scores_final, negatives_scores_final, valid_rows_with_positive_mask

    def _mask_loss(self, loss: tf.Tensor, valid_rows_with_positive_mask: tf.Tensor) -> tf.Tensor:
        """Sets the loss of false negatives to zero

        Parameters
        ----------
        loss : tf.Tensor
            Loss tensor
        valid_rows_with_positive_mask: tf.Tensor
            1D Boolean mask tensor indicating which row contains the positive example (True)
            or not (False). If can be False during evaluation, as the loss is computed among
            the topk and the positive item might not be among the top-k
        valid_negatives_mask : tf.Tensor, optional
            2D Boolean mask tensor matching the dims of `y_pred`, which is False only for positions
            where the the negative item id is equal to the positive item id, by default None

        Returns
        -------
        tf.Tensor
            Loss with zeroed values for false negatives
        """
        # Setting to zero the loss of false negatives and of rows with no positive sample
        loss = loss * tf.cast(tf.expand_dims(valid_rows_with_positive_mask, -1), dtype=loss.dtype)
        if self.valid_negatives_mask is not None:
            loss = loss * tf.cast(self.valid_negatives_mask, dtype=loss.dtype)
        return loss


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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        sub = positives_scores - negatives_scores
        loss = -tf.math.log(add_epsilon_to_zeros(tf.nn.sigmoid(sub)))
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        sub = positives_scores - negatives_scores
        neg_softmax_weights = tf.nn.softmax(negatives_scores, axis=-1)
        reg = tf.square(negatives_scores) * neg_softmax_weights * self.reg_lambda

        loss = -tf.math.log(add_epsilon_to_zeros(tf.nn.sigmoid(sub) * neg_softmax_weights)) + reg
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        sub = negatives_scores - positives_scores
        loss = tf.reduce_mean(
            tf.nn.sigmoid(sub) + tf.nn.sigmoid(tf.square(negatives_scores)),
            keepdims=True,
            axis=-1,
        ) - tf.nn.sigmoid(tf.square(positives_scores)) / tf.cast(
            tf.shape(negatives_scores)[1], tf.float32
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        sub = negatives_scores - positives_scores
        # Equivalent to log(1 + exp(sub))
        loss = tf.nn.relu(sub) + tf.math.log1p(add_epsilon_to_zeros(tf.math.exp(-tf.abs(sub))))
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

    @docstring_parameter(pairwise_losses_compute_docstring=PAIRWISE_LOSSES_COMPUTE_DOCSTRING)
    def compute(self, positives_scores: tf.Tensor, negatives_scores: tf.Tensor) -> tf.Tensor:
        """
        {pairwise_losses_compute_docstring}
        """
        loss = tf.nn.relu(1 + negatives_scores - positives_scores)
        return loss
