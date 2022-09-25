import pytest

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


def test_encoder_block(music_streaming_data: Dataset):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "user_genres"]
    )

    schema = music_streaming_data.schema
    user_schema = schema.select_by_name(["user_id", "user_genres"])
    user_encoder = mm.EncoderBlock(user_schema, mm.MLPBlock([4]), name="query")
    item_schema = schema.select_by_name(["item_id"])
    item_encoder = mm.EncoderBlock(item_schema, mm.MLPBlock([4]), name="candidate")

    model = mm.Model(
        mm.ParallelBlock(user_encoder, item_encoder),
        mm.ContrastiveOutput(item_schema, "in-batch"),
    )

    assert model.blocks[0]["query"] == user_encoder
    assert model.blocks[0]["candidate"] == item_encoder

    testing_utils.model_test(model, music_streaming_data)

    with pytest.raises(Exception) as excinfo:
        user_encoder.compile("adam")
        user_encoder.fit(music_streaming_data)

    assert "This block is not meant to be trained by itself" in str(excinfo.value)

    user_features = testing_utils.get_model_inputs(user_schema, ["user_genres"])
    testing_utils.test_model_signature(user_encoder, user_features, ["output_1"])

    item_features = testing_utils.get_model_inputs(item_schema)
    testing_utils.test_model_signature(item_encoder, item_features, ["output_1"])
