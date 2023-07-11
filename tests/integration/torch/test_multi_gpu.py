import pytest
import pytorch_lightning as pl

import merlin.models.torch as mm


# TODO: This test is not complete because Lightning launches separate processes
# under the hood with the correct environment variables like `LOCAL_RANK`, but
# the pytest stays in the main process and tests only the LOCAL_RANK=0 case.
# Follow-up with proper test that ensures dataloader is working properly with
# e.g. global_rank > 0.
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

        trainer = pl.Trainer(max_epochs=3, devices=2)
        multi_loader = mm.MultiLoader(data, batch_size=2)
        trainer.fit(model, multi_loader)

        # 100 rows total / 2 devices -> 50 rows per device
        # 50 rows / 2 per batch -> 25 steps per device
        assert trainer.num_training_batches == 25

        assert trainer.global_rank == 0  # This should fail for node 1.
