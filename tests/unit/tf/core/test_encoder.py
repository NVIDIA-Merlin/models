import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_encoder_block(music_streaming_data: Dataset):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "user_genres"]
    )

    schema = music_streaming_data.schema
    user_schema = schema.select_by_tag(Tags.USER)
    user_encoder = mm.EncoderBlock(user_schema, mm.MLPBlock([4]))
    item_schema = schema.select_by_tag(Tags.ITEM)
    item_encoder = mm.EncoderBlock(item_schema, mm.MLPBlock([4]))

    model = mm.Model(
        mm.ParallelBlock(dict(query=user_encoder, item=item_encoder)),
        mm.DotProductCategoricalPrediction(schema),
    )

    testing_utils.model_test(model, music_streaming_data)
