import os

import pytest
import pytorch_lightning as pl
from lightning_fabric import Fabric

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

        Fabric().launch()
        trainer = pl.Trainer(max_epochs=1)
        loader = Loader(
            data,
            batch_size=2,
            shuffle=False,
            global_rank=int(os.environ["LOCAL_RANK"]),
            global_size=2,
            device=int(os.environ["LOCAL_RANK"]),
            drop_last=True,
        )
        trainer.fit(model, loader)
