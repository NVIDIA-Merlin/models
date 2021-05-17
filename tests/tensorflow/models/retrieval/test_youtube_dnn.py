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
retrieval_models = pytest.importorskip("merlin_models.tensorflow.models.retrieval")


def test_youtube_dnn(
    tmpdir,
    categorical_columns,
    categorical_features,
    continuous_columns,
    continuous_features,
    items_column,
    items_features,
    item_labels,
):
    # Input Data
    training_data = {
        "categorical": {**categorical_features, **items_features},
        "continuous": continuous_features,
    }
    inference_data = transform_for_inference(training_data)

    training_ds = tf.data.Dataset.from_tensor_slices((training_data, item_labels))
    num_examples = len(training_ds)

    # Model definition
    model_name = "youtube_dnn"

    model = retrieval_models.YouTubeDNN(
        continuous_columns,
        categorical_columns + [items_column],
        embedding_dims=256,
        hidden_dims=[1024, 512, 256],
    )

    model.input_layer.build({})

    item_embeddings = model.input_layer.embedding_tables["items"]

    def sampled_softmax_loss(y_true, y_pred):
        return tf.nn.sampled_softmax_loss(
            weights=item_embeddings,
            biases=tf.zeros((item_embeddings.shape[0],)),
            labels=y_true,
            inputs=y_pred,
            num_sampled=20,
            num_classes=500,
        )

    model.compile("sgd", sampled_softmax_loss)

    # Training
    model.fit(training_ds)

    # Prediction
    predictions = model(inference_data)

    assert predictions.shape == tf.TensorShape((num_examples, 256))
    assert (predictions > 0).numpy().any()
    assert (predictions < 1).numpy().any()

    # Save/load
    model.save(tmpdir / model_name)
    loaded_model = tf.keras.models.load_model(
        tmpdir / model_name, custom_objects={"sampled_softmax_loss": sampled_softmax_loss}
    )

    predictions = loaded_model(inference_data)

    assert predictions.shape == tf.TensorShape((num_examples, 256))
    assert (predictions > 0).numpy().any()
    assert (predictions < 1).numpy().any()
