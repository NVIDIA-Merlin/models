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


def _CustomBinaryPrediction(name, **kwargs):
    return mm.ModelOutput(
        tf.keras.layers.Dense(1, activation="sigmoid"),
        default_loss="binary_crossentropy",
        default_metrics_fn=lambda: (tf.keras.metrics.Precision(name="precision"),),
        target=name,
        **kwargs
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_prediction_block(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([8]),
        _CustomBinaryPrediction("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "loss_batch",
        "precision",
        "regularization_loss",
    }


def test_logits_scaler(ecommerce_data: Dataset):
    import numpy as np
    from tensorflow.keras.utils import set_random_seed

    set_random_seed(42)
    logits_temperature = 2.0
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.BinaryOutput(
            "click",
            to_call=tf.keras.layers.Dense(1, activation=None),
            logits_temperature=logits_temperature,
        ),
    )

    inputs = mm.sample_batch(ecommerce_data, batch_size=10, include_targets=False)
    prediction_1 = model(inputs, testing=True)

    model.last.logits_scaler = LogitsTemperatureScaler(temperature=1)
    prediction_2 = model(inputs, testing=True)

    assert np.allclose(prediction_1.outputs.numpy(), prediction_2.outputs.numpy() / 2.0)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_parallel_outputs(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.ParallelBlock(
            mm.BinaryOutput("click", pre=mm.MLPBlock([4])),
            mm.BinaryOutput("conversion", pre=mm.MLPBlock([4])),
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(list(history.history.keys())) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "click/binary_output_loss",
            "click/binary_output/auc",
            "click/binary_output/recall",
            "click/binary_output/precision",
            "click/binary_output/binary_accuracy",
            "conversion/binary_output/recall",
            "conversion/binary_output/auc",
            "conversion/binary_output/precision",
            "conversion/binary_output/binary_accuracy",
            "conversion/binary_output_loss",
        ]
    )
