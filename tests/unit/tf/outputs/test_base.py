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
from merlin.models.tf.transforms.bias import LogitsTemperatureScaler
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_prediction_block(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        _BinaryPrediction("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "precision",
        "regularization_loss",
    }


def test_logits_scaler(ecommerce_data: Dataset):
    import numpy as np
    from tensorflow.keras.utils import set_random_seed

    set_random_seed(42)
    logits_temperature = 0.5
    model_1 = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        _BinaryPrediction("click", logits_temperature=logits_temperature),
    )

    inputs = mm.sample_batch(ecommerce_data, batch_size=10, include_targets=False)
    prediction_1 = model_1(inputs)

    model_1.blocks[-1].post = LogitsTemperatureScaler(logits_temperature)
    model_1.blocks[-1].logits_temperature = 1
    prediction_2 = model_1(inputs)

    assert np.allclose(prediction_1.numpy(), prediction_2.numpy())


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_parallel_outputs(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.ParallelBlock(
            _BinaryPrediction("click", pre=mm.MLPBlock([4])),
            _BinaryPrediction("conversion", pre=mm.MLPBlock([4])),
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert list(history.history.keys()) == [
        "loss",
        "click/model_output_loss",
        "conversion/model_output_loss",
        "click/model_output/precision",
        "conversion/model_output/precision",
        "regularization_loss",
    ]


def _BinaryPrediction(name, **kwargs):
    return mm.ModelOutput(
        tf.keras.layers.Dense(1, activation="sigmoid"),
        default_loss="binary_crossentropy",
        default_metrics_fn=lambda: (tf.keras.metrics.Precision(name="precision"),),
        target=name,
        **kwargs
    )
