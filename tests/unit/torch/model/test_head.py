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
import torch

import merlin.models.torch as ml


@pytest.mark.parametrize("task", [ml.BinaryClassificationTask, ml.RegressionTask])
def test_simple_heads(torch_tabular_features, torch_tabular_data, task):
    targets = {"target": torch.randint(2, (100,)).float()}

    body = ml.SequentialBlock(torch_tabular_features, ml.MLPBlock([64]))
    head = task("target").to_head(body, torch_tabular_features)

    body_out = body(torch_tabular_data)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize("task", [ml.BinaryClassificationTask, ml.RegressionTask])
@pytest.mark.parametrize("task_block", [None, ml.MLPBlock([32]), ml.MLPBlock([32]).build([-1, 64])])
# @pytest.mark.parametrize("summary", ["last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(torch_tabular_features, torch_tabular_data, task, task_block):
    inputs = torch_tabular_features
    targets = {"target": torch.randint(2, (100,)).float()}

    body = ml.SequentialBlock(inputs, ml.MLPBlock([64]))
    head = task("target", task_block=task_block).to_head(body, inputs)

    body_out = body(torch_tabular_data)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        ml.MLPBlock([32]),
        ml.MLPBlock([32]).build([-1, 64]),
        dict(classification=ml.MLPBlock([16]), regression=ml.MLPBlock([20])),
    ],
)
def test_head_with_multiple_tasks(torch_tabular_features, torch_tabular_data, task_blocks):
    targets = {
        "classification": torch.randint(2, (100,)).float(),
        "regression": torch.randint(2, (100,)).float(),
    }

    body = ml.SequentialBlock(torch_tabular_features, ml.MLPBlock([64]))
    tasks = [
        ml.BinaryClassificationTask("classification", task_name="classification"),
        ml.RegressionTask("regression", task_name="regression"),
    ]
    head = ml.Head(body, tasks, task_blocks=task_blocks)
    optimizer = torch.optim.Adam(head.parameters())

    with torch.set_grad_enabled(mode=True):
        body_out = body(torch_tabular_data)
        loss = head.compute_loss(body_out, targets)
        metrics = head.calculate_metrics(body_out, targets, call_body=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.min() >= 0 and loss.max() <= 1
    assert len(metrics.keys()) == 4
    if task_blocks:
        assert head.task_blocks["classification"][0] != head.task_blocks["regression"][0]

        assert not torch.equal(
            head.task_blocks["classification"][0][0].weight,
            head.task_blocks["regression"][0][0].weight,
        )
