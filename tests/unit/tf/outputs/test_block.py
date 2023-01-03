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

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_output(ecommerce_data: Dataset, run_eagerly: bool):
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([4]),
        mm.OutputBlock(ecommerce_data.schema),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        mm.MLPBlock([32]),
        dict(click=mm.MLPBlock([16]), play_percentage=mm.MLPBlock([20])),
        {
            "click/binary_output": mm.MLPBlock([16]),
            "play_percentage/regression_output": mm.MLPBlock([20]),
        },
    ],
)
def test_model_with_multiple_tasks(music_streaming_data: Dataset, task_blocks, run_eagerly: bool):
    music_streaming_data.schema = music_streaming_data.schema.without("like")

    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=task_blocks)
    model = mm.Model(inputs, mm.MLPBlock([64]), output_block)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

    assert metrics["loss"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "click/binary_output_loss",
            "play_percentage/regression_output_loss",
            "click/binary_output/precision",
            "click/binary_output/recall",
            "click/binary_output/binary_accuracy",
            "click/binary_output/auc",
            "play_percentage/regression_output/root_mean_squared_error",
        ]
    )
    if task_blocks:
        # Checking that task blocks are different for every task
        assert (
            output_block.parallel_dict["click/binary_output"][0]
            != output_block.parallel_dict["play_percentage/regression_output"][0]
        )
