import pytest
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.batch import Batch, sample_batch
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestSelectKeys:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.batch: Batch = sample_batch(music_streaming_data, batch_size=10)
        self.schema: Schema = music_streaming_data.schema
        self.user_schema: Schema = mm.schema.select(self.schema, Tags.USER)

    def test_forward(self):
        select_user = mm.SelectKeys(self.user_schema)
        outputs = module_utils.module_test(select_user, self.batch.features)

        assert select_user.schema == self.user_schema

        for col in {"user_id", "country", "user_age"}:
            assert col in outputs

        assert "user_genres__values" in outputs
        assert "user_genres__offsets" in outputs
        assert select_user != nn.LayerNorm(10)

    def test_select(self):
        select_user = mm.SelectKeys(self.user_schema)

        user_id = Schema([self.user_schema["user_id"]])
        assert select_user.select(ColumnSchema("user_id")).schema == user_id
        assert select_user.select(Tags.USER).schema == self.user_schema

    def test_setup_schema(self):
        select_user = mm.SelectKeys()
        select_user.setup_schema(self.user_schema["user_id"])
        assert select_user.schema == Schema([self.user_schema["user_id"]])
