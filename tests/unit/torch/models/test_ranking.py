import pytest
import pytorch_lightning as pl

import merlin.models.torch as mm
from merlin.dataloader.torch import Loader
from merlin.models.torch.batch import sample_batch
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema


@pytest.mark.parametrize("output_block", [None, mm.BinaryOutput(ColumnSchema("click"))])
class TestDLRMModel:
    def test_train_dlrm_with_lightning_loader(
        self, music_streaming_data, output_block, dim=2, batch_size=16
    ):
        schema = music_streaming_data.schema.select_by_name(
            ["item_id", "user_id", "user_age", "item_genres", "click"]
        )
        music_streaming_data.schema = schema

        model = mm.DLRMModel(
            schema,
            dim=dim,
            bottom_block=mm.MLPBlock([4, 2]),
            top_block=mm.MLPBlock([4, 2]),
            output_block=output_block,
        )

        trainer = pl.Trainer(max_epochs=1, devices=1)

        with Loader(music_streaming_data, batch_size=batch_size) as train_loader:
            model.initialize(train_loader)
            trainer.fit(model, train_loader)

        assert trainer.logged_metrics["train_loss"] > 0.0

        batch = sample_batch(music_streaming_data, batch_size)
        _ = module_utils.module_test(model, batch)


class TestDCNModel:
    @pytest.mark.parametrize("depth", [1, 2])
    @pytest.mark.parametrize("stacked", [True, False])
    @pytest.mark.parametrize("deep_block", [None, mm.MLPBlock([4, 2])])
    def test_train_dcn_with_lightning_trainer(
        self,
        music_streaming_data,
        depth,
        stacked,
        deep_block,
        batch_size=16,
    ):
        schema = music_streaming_data.schema.select_by_name(
            ["item_id", "user_id", "user_age", "item_genres", "click"]
        )
        music_streaming_data.schema = schema

        model = mm.DCNModel(schema, depth=depth, deep_block=deep_block, stacked=stacked)

        trainer = pl.Trainer(max_epochs=1, devices=1)

        with Loader(music_streaming_data, batch_size=batch_size) as train_loader:
            model.initialize(train_loader)
            trainer.fit(model, train_loader)

        assert trainer.logged_metrics["train_loss"] > 0.0

        batch = sample_batch(music_streaming_data, batch_size)
        _ = module_utils.module_test(model, batch)
