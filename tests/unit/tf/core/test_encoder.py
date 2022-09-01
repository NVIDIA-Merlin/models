import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


def test_encoder_block(music_streaming_data: Dataset):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "user_genres"]
    )

    schema = music_streaming_data.schema
    user_schema = schema.select_by_name(["user_id", "user_genres"])
    user_encoder = mm.EncoderBlock(user_schema, mm.MLPBlock([4]))
    item_schema = schema.select_by_name(["item_id"])
    item_encoder = mm.EncoderBlock(item_schema, mm.MLPBlock([4]))

    model = mm.Model(
        mm.ParallelBlock(dict(query=user_encoder, item=item_encoder)),
        mm.DotProductCategoricalPrediction(schema),
    )

    # testing_utils.model_test(model, music_streaming_data)
    model.compile(run_eagerly=True, optimizer="adam")
    model.fit(music_streaming_data, batch_size=50, epochs=1, steps_per_epoch=1)

    user_features = testing_utils.get_model_inputs(user_schema, ["user_genres"])
    testing_utils.test_model_signature(user_encoder, user_features, ["output_1"])

    item_features = testing_utils.get_model_inputs(item_schema)
    testing_utils.test_model_signature(item_encoder, item_features, ["output_1"])
