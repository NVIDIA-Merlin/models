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

from merlin_models.tf.core import Block

from ..utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemsPredictionTopK(Block):
    """
    Block to extract top-k scores from the item-prediction layer output.

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
