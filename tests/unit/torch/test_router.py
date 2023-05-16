import pytest

from merlin.models.torch.router import RouterBlock, SelectKeys
from merlin.schema import Schema


class TestRouterBlock:
    ...


class TestSelectKeys:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.data = music_streaming_data
        self.schema = music_streaming_data.schema
        self.select_keys = SelectKeys(music_streaming_data.schema)

    def test_forward(self):
        ...
