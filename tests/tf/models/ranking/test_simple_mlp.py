# #
# # Copyright (c) 2021, NVIDIA CORPORATION.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
# import pytest
#
# from tests.conftest import transform_for_inference
#
# tf = pytest.importorskip("tensorflow")
# ranking_models = pytest.importorskip("merlin_models.tf.models.ranking")
#
#
# def test_simple_mlp(
#     tmpdir,
#     categorical_columns,
#     categorical_features,
#     continuous_columns,
#     continuous_features,
#     labels,
# ):
#     # Model definition
#     model_name = "simple_mlp"
#
#     model = ranking_models.SimpleMLP(
#         continuous_columns,
#         categorical_columns,
#         embedding_dims=512,
#         hidden_dims=[512, 256, 128],
#     )
#
#     model.compile("sgd", "binary_crossentropy")
#
#     # Input Data
#     training_data = {"mlp": {**continuous_features, **categorical_features}}
#     inference_data = transform_for_inference(training_data)
#
#     # Training
#     training_ds = tf.data.Dataset.from_tensor_slices((training_data, labels))
#     num_examples = len(training_ds)
#
#     model.fit(training_ds)
#
#     # Prediction
#     predictions = model(inference_data)
#
#     assert predictions.shape == tf.TensorShape((num_examples, 1))
#     assert (predictions > 0).numpy().all()
#     assert (predictions < 1).numpy().all()
#
#     # Save/load
#     model.save(tmpdir / model_name)
#     loaded_model = tf.keras.models.load_model(tmpdir / model_name)
#
#     predictions = loaded_model(inference_data)
#
#     assert predictions.shape == tf.TensorShape((num_examples, 1))
#     assert (predictions > 0).numpy().all()
#     assert (predictions < 1).numpy().all()
