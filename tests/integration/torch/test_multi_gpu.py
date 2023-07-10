import pytest
import pytorch_lightning as pl

import merlin.models.torch as mm


class TestMultiGPU:
    @pytest.mark.multigpu
    def test_multi_gpu(self, music_streaming_data):
        schema = music_streaming_data.schema
        data = music_streaming_data
        model = mm.Model(
            mm.TabularInputBlock(schema, init="defaults"),
            mm.MLPBlock([5]),
            mm.BinaryOutput(schema["click"]),
        )

        trainer = pl.Trainer(max_epochs=3, devices=[0, 1])
        multi_loader = mm.MultiLoader(data, batch_size=2)
        trainer.fit(model, multi_loader)

        # 100 rows total / 2 devices -> 50 rows per device
        # 50 rows / 2 per batch -> 25 steps per device
        assert trainer.num_training_batches == 25

        assert trainer.global_rank == 0  # This should fail for node 1.
