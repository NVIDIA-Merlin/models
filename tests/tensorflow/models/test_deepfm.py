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

from tests.conftest import transform_for_inference

tf = pytest.importorskip("tensorflow")
models = pytest.importorskip("merlin_models.tensorflow.models")


def test_deepfm(
    tmpdir,
    categorical_columns,
    categorical_features,
    continuous_columns,
    continuous_features,
    labels,
):
    # Model definition
    model_name = "deepfm"

    model = models.DeepFM(
        continuous_columns,
        categorical_columns,
        embedding_dims=512,
        hidden_dims=[512, 256, 128],
    )

    model.compile("sgd", "binary_crossentropy")

    # Training
    training_data = {"deep": {**continuous_features}, "fm": {**categorical_features}}

    training_ds = tf.data.Dataset.from_tensor_slices((training_data, labels))

    model.fit(training_ds)

    # Batch prediction
    predictions = model.predict(training_data)
    not_nan = tf.math.logical_not(tf.math.is_nan(predictions))

    assert not_nan.numpy().all()
    assert (predictions > 0).all()
    assert (predictions < 1).all()

    # Save/load
    model.save(tmpdir / model_name)
    loaded = tf.keras.models.load_model(tmpdir / model_name)

    # Inference
    infer = loaded.signatures["serving_default"]

    inference_data = transform_for_inference(training_data)

    outputs = infer(**inference_data)
    predictions = outputs["output_1"]
    not_nan = tf.math.logical_not(tf.math.is_nan(predictions))

    assert not_nan.numpy().all()
    assert (predictions > 0).numpy().all()
    assert (predictions < 1).numpy().all()
