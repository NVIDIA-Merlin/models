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
import pytest

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(ecommerce_data: Dataset, run_eagerly):
    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    testing_utils.model_test(model, ecommerce_data)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mf_model_single_binary_task(ecommerce_data, run_eagerly):
    model = ml.MatrixFactorizationModel(
        ecommerce_data.schema,
        dim=64,
        aggregation="cosine",
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model(music_streaming_data, run_eagerly):
    model = ml.DLRMModel(
        music_streaming_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
        top_block=ml.MLPBlock([32]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_without_continuous_features(ecommerce_data, run_eagerly):
    model = ml.DLRMModel(
        ecommerce_data.schema,
        embedding_dim=64,
        top_block=ml.MLPBlock([32]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dcn_model(ecommerce_data, stacked, run_eagerly):
    model = ml.DCNModel(
        ecommerce_data.schema,
        depth=3,
        deep_block=ml.MLPBlock([64]),
        stacked=stacked,
        embedding_options=ml.EmbeddingOptions(
            embedding_dims=None,
            embedding_dim_default=64,
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=2.0,
        ),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_multi_task(music_streaming_data, run_eagerly):
    dlrm_body = ml.DLRMBlock(
        music_streaming_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
        top_block=ml.MLPBlock([32]),
    )

    tasks_blocks = dict(click=ml.MLPBlock([16]), play_percentage=ml.MLPBlock([20]))

    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema, task_blocks=tasks_blocks)

    model = ml.Model(dlrm_body, ml.MLPBlock([64]), prediction_tasks)

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("prediction_task", [ml.BinaryClassificationTask, ml.RegressionTask])
def test_serialization_model(ecommerce_data: Dataset, prediction_task):
    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = ml.Model(body, prediction_task("click"))

    testing_utils.model_test(model, ecommerce_data)
