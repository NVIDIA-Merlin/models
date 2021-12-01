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

if pytorch.cuda.is_available():
    devices = ["cpu", "cuda"]
else:
    devices = ["cpu"]


def test_simple_model(torch_tabular_features, torch_tabular_data):
    targets = {"target": pytorch.randint(2, (100,)).float()}

    inputs = torch_tabular_features
    body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))
    model = tr.BinaryClassificationTask("target").to_model(body, inputs)

    dataset = [(torch_tabular_data, targets)]
    losses = model.fit(dataset, num_epochs=5)
    metrics = model.evaluate(dataset, mode="eval")

    # assert list(metrics.keys()) == ["precision", "recall", "accuracy"]
    assert len(metrics) == 3
    assert len(losses) == 5
    assert all(loss.min() >= 0 and loss.max() <= 1 for loss in losses)


# def test_model_with_multiple_heads_and_tasks(
#     yoochoose_schema,
#     torch_yoochoose_tabular_transformer_features,
#     torch_yoochoose_like,
# ):
#     # Tabular classification and regression tasks
#     targets = {
#         "classification": pytorch.randint(2, (100,)).float(),
#         "regression": pytorch.randint(2, (100,)).float(),
#     }
#
#     non_sequential_features_schema = yoochoose_schema.select_by_name(["user_age", "user_country"])
#
#     tabular_features = tr.TabularFeatures.from_schema(
#         non_sequential_features_schema,
#         max_sequence_length=20,
#         continuous_projection=64,
#         aggregation="concat",
#     )
#
#     body = tr.SequentialBlock(tabular_features, tr.MLPBlock([64]))
#     tasks = [
#         tr.BinaryClassificationTask("classification"),
#         tr.RegressionTask("regression"),
#     ]
#     head_1 = tr.Head(body, tasks)
#
#     # Session-based classification and regression tasks
#     targets_2 = {
#         "classification_session": pytorch.randint(2, (100,)).float(),
#         "regression_session": pytorch.randint(2, (100,)).float(),
#     }
#     transformer_config = tconf.XLNetConfig.build(
#         d_model=64, n_head=4, n_layer=2, total_seq_length=20
#     )
#     body_2 = tr.SequentialBlock(
#         torch_yoochoose_tabular_transformer_features,
#         tr.MLPBlock([64]),
#         tr.TransformerBlock(transformer_config),
#     )
#     tasks_2 = [
#         tr.BinaryClassificationTask("classification_session", summary_type="last"),
#         tr.RegressionTask("regression_session", summary_type="mean"),
#     ]
#     head_2 = tr.Head(body_2, tasks_2)
#
#     # Final prediction with two heads
#     prediction = tr.Model(head_1, head_2)
#
#     # launch training
#     targets.update(targets_2)
#     dataset = [(torch_yoochoose_like, targets)]
#     losses = prediction.fit(dataset, num_epochs=5)
#     metrics = prediction.evaluate(dataset)
#
#     assert list(metrics.keys()) == [
#         "eval_classification/binary_classification_task",
#         "eval_regression/regression_task",
#         "eval_classification_session/binary_classification_task",
#         "eval_regression_session/regression_task",
#     ]
#     assert len(losses) == 5
#     assert all(loss is not None for loss in losses)


def test_multi_head_model_wrong_weights(torch_tabular_features, torch_yoochoose_like):
    with pytest.raises(ValueError) as excinfo:
        inputs = torch_tabular_features
        body = tr.SequentialBlock(inputs, tr.MLPBlock([64]))

        head_1 = tr.BinaryClassificationTask("classification").to_head(body, inputs)
        head_2 = tr.RegressionTask("regression").to_head(body, inputs)

        tr.Model(head_1, head_2, head_weights=[0.4])

    assert "`head_weights` needs to have the same length " "as the number of heads" in str(
        excinfo.value
    )
