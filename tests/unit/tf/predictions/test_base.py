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
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_prediction_block(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([64]),
        _BinaryPrediction("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {"loss", "precision", "regularization_loss"}


@pytest.mark.parametrize("run_eagerly", [True])
def test_parallel_prediction_blocks(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([64]),
        mm.ParallelBlock(
            _BinaryPrediction("click", pre=mm.MLPBlock([16])),
            _BinaryPrediction("conversion", pre=mm.MLPBlock([16])),
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert len(history.history.keys()) == 6


def _BinaryPrediction(name, **kwargs):
    return mm.PredictionBlock(
        tf.keras.layers.Dense(1, activation="sigmoid"),
        default_loss="binary_crossentropy",
        default_metrics=(tf.keras.metrics.Precision(),),
        target=name,
        **kwargs
    )
