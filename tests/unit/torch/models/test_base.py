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
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, Precision, Recall

import merlin.models.torch as mm
from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch, sample_batch
from merlin.models.torch.models.base import compute_loss
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema


class PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1


class TimeTwo(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 2


class TestModel:
    def test_init_default(self):
        model = mm.Model(mm.Block(), nn.Linear(10, 10))
        assert isinstance(model, mm.Model)
        assert len(model) == 2
        assert isinstance(model.configure_optimizers()[0], torch.optim.Adam)

    def test_init_optimizer_and_scheduler(self):
        model = mm.Model(mm.MLPBlock([4, 4]))
        model.initialize(mm.Batch(torch.rand(2, 2)))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        opt, sched = model.configure_optimizers(optimizer, scheduler)
        assert opt == [optimizer]
        assert sched == [scheduler]

    def test_pre_and_pre(self):
        inputs = torch.tensor([[1, 2], [3, 4]])
        model = mm.Model(mm.Block(), mm.Block())
        assert torch.equal(model(inputs), inputs)

        model.prepend(PlusOne())
        assert torch.equal(model(inputs), inputs + 1)

        model.append(TimeTwo())
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

        with pytest.raises(ValueError):
            mm.Model(mm.Block(), pre=mm.Block())

    def test_script(self):
        model = mm.Model(mm.Block(), mm.Block())
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.initialize(mm.Batch(inputs, None))

        outputs = module_utils.module_test(
            model.to_torchscript(method="script"), inputs, schema_trace=False
        )

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
        expected_loss = nn.BCELoss()(expected_outputs, targets["target"])
        assert torch.allclose(loss, expected_loss)

    def test_step_with_dataloader(self):
        model = mm.Model(
            mm.Concat(),
            mm.BinaryOutput(ColumnSchema("target")),
        )

        feature = [2.0, 3.0]
        target = [0.0, 1.0]
        dataset = Dataset(pd.DataFrame({"feature": feature, "target": target}))

        with Loader(dataset, batch_size=2) as loader:
            model.initialize(loader)
            batch = loader.peek()

        loss = model.training_step(batch, 0)
        assert loss > 0.0
        assert torch.equal(
            model.validation_step(batch, 0)["loss"], model.test_step(batch, 0)["loss"]
        )

    def test_step_with_batch(self):
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
        assert torch.equal(
            model.validation_step(batch, 0)["loss"], model.test_step(batch, 0)["loss"]
        )

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

    def test_output_schema(self):
        model = mm.Model(mm.Block())
        inputs = {
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
        outputs = mm.schema.trace(model, inputs)
        schema = mm.output_schema(model)
        for name in outputs:
            assert name in schema.column_names
            assert schema[name].dtype.name == str(outputs[name].dtype).split(".")[-1]

    def test_no_output_schema(self):
        model = mm.Model(PlusOne())
        with pytest.raises(ValueError, match="Could not get output schema of PlusOne()"):
            mm.output_schema(model)

    def test_train_classification_with_lightning_trainer(self, music_streaming_data, batch_size=16):
        schema = music_streaming_data.schema.select_by_name(
            ["item_id", "user_id", "user_age", "item_genres", "click"]
        )
        music_streaming_data.schema = schema

        model = mm.Model(
            mm.TabularInputBlock(schema, init="defaults"),
            mm.MLPBlock([4, 2]),
            mm.BinaryOutput(schema.select_by_name("click").first),
        )

        trainer = pl.Trainer(max_epochs=1, devices=1)
        trainer.fit(model, Loader(music_streaming_data, batch_size=batch_size))

        assert trainer.logged_metrics["train_loss"] > 0.0
        assert trainer.num_training_batches == 7  # 100 rows // 16 per batch + 1 for last batch

        batch = sample_batch(music_streaming_data, batch_size)
        _ = module_utils.module_test(model, batch)


class TestMultiLoader:
    def test_train_dataset(self, music_streaming_data):
        multi_loader = mm.MultiLoader(music_streaming_data)
        assert multi_loader.train_dataloader() is multi_loader.loader_train

    def test_train_loader(self, music_streaming_data):
        multi_loader = mm.MultiLoader(Loader(music_streaming_data, 2))
        assert multi_loader.train_dataloader() is multi_loader.loader_train

    def test_valid_dataloader(self, music_streaming_data):
        multi_loader = mm.MultiLoader(music_streaming_data, music_streaming_data)
        assert multi_loader.val_dataloader() is multi_loader.loader_valid

    def test_test_dataloader(self, music_streaming_data):
        multi_loader = mm.MultiLoader(*([music_streaming_data] * 3))
        assert multi_loader.test_dataloader() is multi_loader.loader_test

    def test_teardown(self, music_streaming_data):
        multi_loader = mm.MultiLoader(*([music_streaming_data] * 3))
        multi_loader.teardown(None)
        assert not hasattr(multi_loader, "loader_train")
        assert not hasattr(multi_loader, "loader_valid")
        assert not hasattr(multi_loader, "loader_test")


class TestComputeLoss:
    def test_tensor_inputs(self):
        predictions = torch.sigmoid(torch.randn(2, 1))
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput(ColumnSchema("a"))]
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.BCELoss()(predictions, targets)
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
        predictions = torch.sigmoid(torch.randn(2, 1))
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = [mm.BinaryOutput(ColumnSchema("a"))]
        results = compute_loss(predictions, targets, model_outputs, compute_metrics=False)
        assert sorted(results.keys()) == ["loss"]

    def test_dict_inputs(self):
        outputs = mm.ParallelBlock({"a": mm.BinaryOutput(ColumnSchema("a"))})
        predictions = outputs(torch.randn(2, 1))
        targets = {"a": torch.randint(2, (2, 1), dtype=torch.float32)}

        results = compute_loss(predictions, targets, outputs.find(mm.ModelOutput))
        expected_loss = nn.BCELoss()(predictions["a"], targets["a"])
        assert torch.allclose(results["loss"], expected_loss)

    def test_mixed_inputs(self):
        predictions = {"a": torch.randn(2, 1)}
        targets = torch.randint(2, (2, 1), dtype=torch.float32)
        model_outputs = (mm.RegressionOutput(ColumnSchema("a")),)
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.MSELoss()(predictions["a"], targets)
        assert torch.allclose(results["loss"], expected_loss)

    def test_single_model_output(self):
        predictions = {"foo": torch.randn(2, 1)}
        targets = {"foo": torch.randint(2, (2, 1), dtype=torch.float32)}
        model_outputs = [mm.RegressionOutput(ColumnSchema("foo"))]
        results = compute_loss(predictions, targets, model_outputs)
        expected_loss = nn.MSELoss()(predictions["foo"], targets["foo"])
        assert torch.allclose(results["loss"], expected_loss)

    def test_tensor_input_no_targets(self):
        predictions = torch.randn(2, 1)
        binary_output = mm.RegressionOutput(ColumnSchema("foo"))
        results = compute_loss(predictions, None, (binary_output,))
        expected_loss = nn.MSELoss()(predictions, torch.zeros(2, 1))
        assert torch.allclose(results["loss"], expected_loss)

    def test_dict_input_no_targets(self):
        predictions = {"foo": torch.randn(2, 1)}
        binary_output = mm.RegressionOutput(ColumnSchema("foo"))
        results = compute_loss(predictions, None, (binary_output,))
        expected_loss = nn.MSELoss()(predictions["foo"], torch.zeros(2, 1))
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
