#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import pandas as pd
import pytest
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, Precision, Recall

import merlin.models.torch as mm
from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch
from merlin.models.torch.models.base import compute_loss
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema


class PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1


class TimeTwo(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 2


class TestModel:
    def test_init_default(self):
        model = mm.Model(mm.Block(), mm.Block())
        assert isinstance(model, mm.Model)
        assert len(model.blocks) == 2
        assert model.schema is None
        assert model.optimizer is torch.optim.Adam

    def test_init_schema(self):
        schema = Schema([ColumnSchema("foo")])
        model = mm.Model(mm.Block(), mm.Block(), schema=schema)
        assert len(model.blocks) == 2
        assert model.schema.first.name == "foo"

    def test_init_optimizer(self):
        optimizer = torch.optim.SGD
        model = mm.Model(mm.Block(), mm.Block(), optimizer=optimizer)
        assert model.optimizer is torch.optim.SGD

    def test_pre_and_pre(self):
        inputs = torch.tensor([[1, 2], [3, 4]])
        model = mm.Model(mm.Block(), mm.Block())
        assert torch.equal(model(inputs), inputs)

        model.pre.append(PlusOne())
        assert torch.equal(model(inputs), inputs + 1)

        model.post.append(TimeTwo())
        assert torch.equal(model(inputs), (inputs + 1) * 2)

    def test_initialize_with_dataset(self):
        dataset = Dataset(
            pd.DataFrame(
                {
                    "feature": [1, 2, 3],
                    "target": [1, 0, 0],
                }
            )
        )
        model = mm.Model(mm.Block(), mm.Block())
        model.initialize(dataset)

    def test_initialize_with_dataloader(self):
        dataset = Dataset(
            pd.DataFrame(
                {
                    "feature": [1, 2, 3],
                    "target": [1, 0, 0],
                }
            )
        )
        model = mm.Model(mm.Block(), mm.Block())
        with Loader(dataset, batch_size=1) as loader:
            model.initialize(loader)

    def test_initialize_raises_error(self):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model = mm.Model(mm.Block(), mm.Block())
        with pytest.raises(RuntimeError, match="Unexpected input type"):
            model.initialize(inputs)

    def test_script(self):
        model = mm.Model(mm.Block(), mm.Block())
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.initialize(mm.Batch(inputs, None))

        outputs = module_utils.module_test(model.to_torchscript(method="script"), inputs)

        assert torch.equal(inputs, outputs)

    def test_trace(self):
        model = mm.Model(mm.Block(), mm.Block())
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(
            model.to_torchscript(method="trace", example_inputs=inputs), inputs, method="trace"
        )

        assert torch.equal(inputs, outputs)

    def test_training_step_values(self):
        model = mm.Model(
            mm.Concat(),
            mm.BinaryOutput(ColumnSchema("target")),
        )
        features = {"feature": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        targets = {"target": torch.tensor([[0.0], [1.0]])}
        loss = model.training_step((features, targets), 0)
        (weights, bias) = model.parameters()
        expected_outputs = nn.Sigmoid()(torch.matmul(features["feature"], weights.T) + bias)
        expected_loss = nn.BCEWithLogitsLoss()(expected_outputs, targets["target"])
        assert torch.allclose(loss, expected_loss)

    def test_training_step_with_dataloader(self):
        model = mm.Model(
            mm.Concat(),
            mm.BinaryOutput(ColumnSchema("target")),
        )
        feature = [[1.0, 2.0], [3.0, 4.0]]
        target = [[0.0], [1.0]]
        dataset = Dataset(pd.DataFrame({"feature": feature, "target": target}))
        with Loader(dataset, batch_size=1) as loader:
            model.initialize(loader)
            batch = loader.peek()

        loss = model.training_step(batch, 0)
        assert loss > 0.0

    def test_training_step_with_batch(self):
        model = mm.Model(
            mm.Concat(),
            mm.BinaryOutput(ColumnSchema("target")),
        )
        feature = {"feature": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        target = {"target": torch.tensor([[0.0], [1.0]])}
        batch = Batch(feature, target)
        model.initialize(batch)
        loss = model.training_step(batch, 0)
        assert loss > 0.0

    def test_training_step_missing_output(self):
        model = mm.Model(mm.Block())
        features = {"feature": torch.tensor([1, 2])}
        targets = {"target": torch.tensor([0, 1])}
        with pytest.raises(RuntimeError, match="No model outputs found"):
            _ = model.training_step((features, targets), 0)

    def test_model_outputs(self):
        block1 = mm.RegressionOutput(ColumnSchema("foo"))
        block2 = mm.BinaryOutput(ColumnSchema("bar"))
        model = mm.Model(
            mm.Block(),
            mm.MLPBlock([4, 2]),
            mm.ParallelBlock({"block1": block1, "block2": block2}),
        )
        assert len(model.model_outputs()) == 2
        assert block1 in model.model_outputs()
        assert block2 in model.model_outputs()

    def test_first(self):
        model = mm.Model(mm.Block(name="a"), mm.Block(name="b"), mm.Block(name="c"))
        assert model.first()._name == "a"

    def test_last(self):
        model = mm.Model(mm.Block(name="a"), mm.Block(name="b"), mm.Block(name="c"))
        assert model.last()._name == "c"

    def test_input_schema(self):
        schema = Schema([ColumnSchema("foo"), ColumnSchema("bar")])
        model = mm.Model(mm.Block(), mm.Block(), schema=schema)
        assert model.input_schema() == schema

    def test_no_input_schema(self):
        model = mm.Model(mm.Block(), mm.Block())
        assert model.input_schema() == Schema([])

    def test_output_schema(self):
        model = mm.Model(mm.Block())
        inputs = {
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
        outputs = model(inputs)
        schema = model.output_schema()
        for name in outputs:
            assert name in schema.column_names
            assert schema[name].dtype.name == str(outputs[name].dtype).split(".")[-1]

    def test_no_output_schema(self):
        model = mm.Model(PlusOne())
        with pytest.raises(RuntimeError, match="No output schema found"):
            _ = model.output_schema()

    # def test_train_classification(self, music_streaming_data):
    #     schema = music_streaming_data.schema.without(["user_genres", "like", "item_genres"])
    #     music_streaming_data.schema = schema

    #     model = mm.Model(
    #        mm.TabularInputBlock(schema),
    #        mm.MLPBlock([4, 2]),
    #        mm.BinaryOutput(schema.select_by_name("click").first),
    #        schema=schema,
    #     )

    #     trainer = pl.Trainer(max_epochs=1)

    #     with Loader(music_streaming_data, batch_size=16) as loader:
    #         model.initialize(loader)
    #         trainer.fit(model, loader)


class TestComputeLoss:
    def test_tensor_inputs(self):
        predictions = torch.randn(2, 1)
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput(ColumnSchema("a"))]
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.BCEWithLogitsLoss()(predictions, targets)
        expected_auroc = AUROC(task="binary")(predictions, targets)
        expected_acc = Accuracy(task="binary")(predictions, targets)
        expected_prec = Precision(task="binary")(predictions, targets)
        expected_rec = Recall(task="binary")(predictions, targets)

        assert isinstance(results, dict)
        assert sorted(results.keys()) == [
            "binary_accuracy",
            "binary_auroc",
            "binary_precision",
            "binary_recall",
            "loss",
        ]
        assert torch.allclose(results["loss"], expected_loss)
        assert torch.allclose(results["binary_auroc"], expected_auroc)
        assert torch.allclose(results["binary_accuracy"], expected_acc)
        assert torch.allclose(results["binary_precision"], expected_prec)
        assert torch.allclose(results["binary_recall"], expected_rec)

    def test_no_metrics(self):
        predictions = torch.randn(2, 1)
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput(ColumnSchema("a"))]
        results = compute_loss(predictions, targets, model_outputs, compute_metrics=False)
        assert sorted(results.keys()) == ["loss"]

    def test_dict_inputs(self):
        predictions = {"a": torch.randn(2, 1)}
        targets = {"a": torch.randint(2, (2, 1), dtype=torch.float32)}
        model_outputs = (mm.BinaryOutput(ColumnSchema("a")),)
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.BCEWithLogitsLoss()(predictions["a"], targets["a"])
        assert torch.allclose(results["loss"], expected_loss)

    def test_mixed_inputs(self):
        predictions = {"a": torch.randn(2, 1)}
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = (mm.BinaryOutput(ColumnSchema("a")),)
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.BCEWithLogitsLoss()(predictions["a"], targets)
        assert torch.allclose(results["loss"], expected_loss)

    def test_single_model_output(self):
        predictions = {"foo": torch.randn(2, 1)}
        targets = {"foo": torch.randint(2, (2, 1), dtype=torch.float32)}
        model_outputs = [mm.BinaryOutput(ColumnSchema("foo"))]
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.BCEWithLogitsLoss()(predictions["foo"], targets["foo"])
        assert torch.allclose(results["loss"], expected_loss)

    def test_tensor_input_no_targets(self):
        predictions = torch.randn(2, 1)
        binary_output = mm.BinaryOutput(ColumnSchema("foo"))
        results = compute_loss(predictions, None, (binary_output,))
        expected_loss = nn.BCEWithLogitsLoss()(predictions, torch.zeros(2, 1))
        assert torch.allclose(results["loss"], expected_loss)

    def test_dict_input_no_targets(self):
        predictions = {"foo": torch.randn(2, 1)}
        binary_output = mm.BinaryOutput(ColumnSchema("foo"))
        results = compute_loss(predictions, None, (binary_output,))
        expected_loss = nn.BCEWithLogitsLoss()(predictions["foo"], torch.zeros(2, 1))
        assert torch.allclose(results["loss"], expected_loss)

    def test_no_target_raises_error(self):
        predictions = torch.randn(2, 1)
        binary_output = mm.BinaryOutput(ColumnSchema("foo"))
        delattr(binary_output, "target")
        with pytest.raises(ValueError, match="has no target"):
            _ = compute_loss(predictions, None, (binary_output,))

    def test_unknown_targets_type(self):
        predictions = torch.randn(2, 1)
        targets = [torch.randint(2, (2, 1), dtype=torch.float32)]
        model_outputs = [mm.BinaryOutput(ColumnSchema("foo"))]
        with pytest.raises(ValueError, match="Unknown 'targets' type"):
            _ = compute_loss(predictions, targets, model_outputs)

    def test_unknown_predictions_type(self):
        predictions = [torch.randn(2, 1)]
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput(ColumnSchema("foo"))]
        with pytest.raises(ValueError, match="Unknown 'predictions' type"):
            _ = compute_loss(predictions, targets, model_outputs)

    def test_target_column_not_in_inputs(self):
        predictions = {"foo": torch.randn(2, 1)}
        targets = {"bar": torch.randint(2, (2, 1), dtype=torch.float32)}
        model_outputs = [mm.BinaryOutput(ColumnSchema("bar"))]
        with pytest.raises(RuntimeError, match="Column 'bar' not found"):
            _ = compute_loss(predictions, targets, model_outputs)
