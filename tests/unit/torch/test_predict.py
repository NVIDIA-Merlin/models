import pandas as pd
import pytest
import torch
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.predict import Encoder, Predictor
from merlin.models.torch.utils.schema_utils import SchemaTrackingMixin
from merlin.schema import Tags


class TensorOutputModel(nn.Module):
    def forward(self, x):
        return x["position"] * 2


class DictOutputModel(SchemaTrackingMixin, nn.Module):
    def __init__(self, output_name: str = "testing"):
        super().__init__()
        self.name = output_name

    def forward(self, x):
        return {self.name: x["user_id"] * 2}


class TestEncoder:
    def test_loader(self, music_streaming_data):
        loader = Loader(music_streaming_data, batch_size=10)

        ddf = music_streaming_data.to_ddf()
        num_items = ddf["item_id"].nunique().compute()

        encoder = Encoder(TensorOutputModel())
        outputs = encoder(loader, index=Tags.ITEM_ID).compute()

        assert outputs.index.name == "item_id"
        assert len(outputs) == num_items

    def test_dataset(self, music_streaming_data):
        encoder = Encoder(DictOutputModel(), selection=Tags.USER)

        with pytest.raises(ValueError):
            encoder(music_streaming_data)

        output = encoder(music_streaming_data, batch_size=10, index=Tags.USER_ID, unique=False)
        output_df = output.compute()
        assert len(output_df) == 100
        assert set(output.schema.column_names) == {"testing"}
        assert output_df.index.name == "user_id"

    def test_tensor_dict(self):
        encoder = Encoder(TensorOutputModel())
        outputs = encoder({"position": torch.tensor([1, 2, 3, 4])})

        assert len(outputs) == 4
        assert hasattr(outputs, "columns")

    def test_tensor(self):
        encoder = Encoder(nn.Identity())
        outputs = encoder(torch.tensor([1, 2, 3, 4]))

        assert len(outputs) == 4
        assert hasattr(outputs, "columns")

    def test_df(self):
        encoder = Encoder(nn.Identity())
        outputs = encoder(pd.DataFrame({"a": [1, 2, 3, 4]}))

        assert len(outputs) == 4
        assert hasattr(outputs, "columns")

    def test_exceptions(self):
        encoder = Encoder(DictOutputModel())

        with pytest.raises(ValueError):
            encoder("")

        with pytest.raises(ValueError):
            encoder.encode_dataset(torch.tensor([1, 2, 3]))


class TestPredictor:
    def test_dataset(self, music_streaming_data):
        predictor = Predictor(DictOutputModel("click"), prediction_suffix="_")
        output = predictor(music_streaming_data, batch_size=10)
        output_df = output.compute()
        assert len(output_df) == 100

        for col in music_streaming_data.schema.column_names:
            assert col in output_df.columns

        assert "click_" in output_df.columns
        assert "click_" in output.schema.column_names
        assert len(output_df.columns) == len(output.schema)

    def test_no_targets(self):
        predictor = Predictor(TensorOutputModel())

        df = pd.DataFrame({"position": [1, 2, 3, 4]})
        outputs = predictor(Dataset(df), batch_size=2)
        output_df = outputs.compute()

        assert len(outputs.schema) == 2
        assert hasattr(output_df, "columns")

    def test_no_targets_dict(self):
        predictor = Predictor(DictOutputModel())

        df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
        outputs = predictor(Dataset(df), batch_size=2)
        output_df = outputs.compute()

        assert len(outputs.schema) == 2
        assert hasattr(output_df, "columns")
