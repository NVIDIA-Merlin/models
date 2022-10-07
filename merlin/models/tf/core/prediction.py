#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from typing import Dict, NamedTuple, Optional, Union

import tensorflow as tf

from merlin.core.dispatch import DataFrameType, get_lib

TensorLike = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class PredictionContext(NamedTuple):
    features: Dict[str, TensorLike]
    targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None
    mask: tf.Tensor = (None,)
    training: bool = False
    testing: bool = False

    def with_updates(
        self, targets=None, features=None, mask=None, training=None, testing=None
    ) -> "PredictionContext":
        return PredictionContext(
            features if features is not None else self.features,
            targets if targets is not None else self.targets,
            mask if mask is not None else self.mask,
            training or self.training,
            testing or self.testing,
        )

    def to_call_dict(self):
        outputs = {
            "features": self.features,
            "training": self.training,
            "testing": self.testing,
        }

        if self.training or self.testing:
            outputs["mask"] = self.mask
            outputs["targets"] = self.targets

        return outputs


class Prediction(NamedTuple):
    outputs: Dict[str, TensorLike]
    targets: Optional[Union[tf.Tensor, Dict[str, tf.Tensor]]] = None
    sample_weight: Optional[tf.Tensor] = None
    features: Optional[Dict[str, TensorLike]] = None

    @property
    def predictions(self):
        return self.outputs


class TopKPrediction(NamedTuple):
    scores: tf.Tensor
    identifiers: tf.Tensor

    def to_df(self) -> DataFrameType:
        score_names = [f"score_{i}" for i in range(self.scores.shape[1])]
        id_names = [f"id_{i}" for i in range(self.identifiers.shape[1])]

        rows = []
        for batch_i in range(self.scores.shape[0]):
            row = {}
            for k in range(self.scores.shape[1]):
                row[score_names[k]] = encode_output(self.scores[batch_i, k])
                row[id_names[k]] = encode_output(self.identifiers[batch_i, k])
            rows.append(row)

        return get_lib().DataFrame(rows)

    @staticmethod
    def output_names(k: int):
        score_names = [f"score_{i}" for i in range(k)]
        id_names = [f"id_{i}" for i in range(k)]

        return score_names + id_names


def encode_output(output: tf.Tensor):
    if len(output.shape) == 2 and output.shape[1] == 1:
        output = tf.squeeze(output)

    return output.numpy()
