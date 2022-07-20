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
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mf_model_single_binary_task(ecommerce_data, run_eagerly):
    model = ml.MatrixFactorizationModel(
        ecommerce_data.schema,
        dim=4,
        aggregation="cosine",
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    model = ml.DLRMModel(
        music_streaming_data.schema,
        embedding_dim=2,
        bottom_block=ml.MLPBlock([2]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    loaded_model, _ = testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)

    features = testing_utils.get_model_inputs(
        music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    )
    testing_utils.test_model_signature(loaded_model, features, ["click/binary_classification_task"])


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_without_continuous_features(ecommerce_data, run_eagerly):
    ecommerce_data.schema = ecommerce_data.schema.select_by_name(
        ["user_categories", "item_category", "click"]
    )
    model = ml.DLRMModel(
        ecommerce_data.schema,
        embedding_dim=2,
        top_block=ml.MLPBlock([2]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dcn_model(music_streaming_data, stacked, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click"]
    )
    model = ml.DCNModel(
        music_streaming_data.schema,
        depth=1,
        deep_block=ml.MLPBlock([2]),
        stacked=stacked,
        embedding_options=ml.EmbeddingOptions(
            embedding_dims=None,
            embedding_dim_default=2,
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=0.2,
        ),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_dlrm_model_multi_task(music_streaming_data, run_eagerly):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["item_id", "user_age", "click", "play_percentage"]
    )
    tasks_blocks = dict(click=ml.MLPBlock([2]), play_percentage=ml.MLPBlock([2]))
    model = ml.Model(
        ml.DLRMBlock(
            music_streaming_data.schema,
            embedding_dim=2,
            bottom_block=ml.MLPBlock([2]),
            top_block=ml.MLPBlock([2]),
        ),
        ml.MLPBlock([2]),
        ml.PredictionTasks(music_streaming_data.schema, task_blocks=tasks_blocks),
    )

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("prediction_task", [ml.BinaryClassificationTask, ml.RegressionTask])
def test_serialization_model(ecommerce_data: Dataset, prediction_task):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema), ml.MLPBlock([2]), prediction_task("click")
    )

    testing_utils.model_test(model, ecommerce_data, reload_model=True)
