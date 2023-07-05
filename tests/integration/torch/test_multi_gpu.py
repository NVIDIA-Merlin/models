import pytest
import pytorch_lightning as pl

import merlin.models.torch as mm
from merlin.loader.torch import Loader


class TestMultiGPU:
    @pytest.mark.multigpu
    def test_multi_gpu(self, music_streaming_data):
        schema = music_streaming_data.schema
        data = music_streaming_data.repartition(2)
        model = mm.Model(
            mm.TabularInputBlock(schema, init="defaults"),
            mm.MLPBlock([5]),
            mm.BinaryOutput(schema["click"]),
        )
        model.initialize(music_streaming_data)

        trainer = pl.Trainer(max_epochs=3, devices=[0, 1])
        loader = Loader(
            data,
            batch_size=2,
            shuffle=False,
            global_size=2,
            drop_last=True,
        )
        trainer.fit(model, loader)
