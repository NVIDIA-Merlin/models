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

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("merlin_models.torch")


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
def test_simple_heads(torch_tabular_features, torch_tabular_data, task):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    body = tr.SequentialBlock(torch_tabular_features, tr.MLPBlock([64]))
    head = task("target").to_head(body, torch_tabular_features)

    body_out = body(torch_tabular_data)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize("task", [tr.BinaryClassificationTask, tr.RegressionTask])
@pytest.mark.parametrize("task_block", [None, tr.MLPBlock([32]), tr.MLPBlock([32]).build([-1, 64])])
# @pytest.mark.parametrize("summary", ["last", "first", "mean", "cls_index"])
def test_simple_heads_on_sequence(torch_tabular_features, torch_yoochoose_like, task, task_block):
    inputs = torch_tabular_features
    targets = {"target": pytorch.randint(2, (100,)).float()}

    body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))
    head = task("target", task_block=task_block).to_head(body, inputs)

    body_out = body(torch_yoochoose_like)
    loss = head.compute_loss(body_out, targets)

    assert loss.min() >= 0 and loss.max() <= 1


@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        tr.MLPBlock([32]),
        tr.MLPBlock([32]).build([-1, 64]),
        dict(classification=tr.MLPBlock([16]), regression=tr.MLPBlock([20])),
    ],
)
def test_head_with_multiple_tasks(torch_tabular_features, torch_tabular_data, task_blocks):
    targets = {
        "classification": pytorch.randint(2, (100,)).float(),
        "regression": pytorch.randint(2, (100,)).float(),
    }

    body = tr.SequentialBlock(torch_tabular_features, tr.MLPBlock([64]))
    tasks = [
        tr.BinaryClassificationTask("classification", task_name="classification"),
        tr.RegressionTask("regression", task_name="regression"),
    ]
    head = tr.Head(body, tasks, task_blocks=task_blocks)
    optimizer = pytorch.optim.Adam(head.parameters())

    with pytorch.set_grad_enabled(mode=True):
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

        assert not pytorch.equal(
            head.task_blocks["classification"][0][0].weight,
            head.task_blocks["regression"][0][0].weight,
        )
