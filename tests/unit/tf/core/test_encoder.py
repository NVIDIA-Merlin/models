import tempfile

import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.core.encoder import TopKEncoder
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


def test_topk_encoder(music_streaming_data: Dataset):
    # TODO: Simplify the test after RetrievalModelV2 is merged
    TOP_K = 50
    NUM_CANDIDATES = 100
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "country", "user_age"]
    )

    # 1. Train a retrieval model
    schema = music_streaming_data.schema
    user_schema = schema.select_by_name(["user_id", "country", "user_age"])
    user_encoder = mm.EncoderBlock(user_schema, mm.MLPBlock([4]), name="query")
    item_schema = schema.select_by_name(["item_id"])
    item_encoder = mm.EncoderBlock(item_schema, mm.MLPBlock([4]), name="candidate")
    retrieval_model = mm.Model(
        mm.ParallelBlock(user_encoder, item_encoder),
        mm.ContrastiveOutput(item_schema, "in-batch"),
    )
    testing_utils.model_test(retrieval_model, music_streaming_data)

    # 2. Get candidates embeddings for the top-k encoder
    _candidate_embeddings = tf.random.uniform(shape=(NUM_CANDIDATES, 4), dtype=tf.float32)

    # 3. Set data-loader for top-k recommendation
    def _item_id_as_target(inputs, targets):
        targets = tf.squeeze(inputs["item_id"])
        return inputs, targets

    loader = mm.Loader(music_streaming_data, batch_size=32, transform=_item_id_as_target)
    batch = next(iter(loader))

    # 4. Define the top-k encoder
    topk_encoder = TopKEncoder(
        query_encoder=retrieval_model.blocks[0]["query"], candidates=_candidate_embeddings, k=TOP_K
    )
    # 5. Get top-k predictions
    batch_output = topk_encoder(batch[0])
    predict_output = topk_encoder.predict(loader)
    assert list(batch_output.scores.shape) == [32, TOP_K]
    assert list(predict_output.scores.shape) == [100, TOP_K]

    # 6. Compute top-k evaluation metrics
    topk_encoder.compile()
    topk_evaluation_metrics = topk_encoder.evaluate(loader, return_dict=True)
    assert set(topk_evaluation_metrics.keys()) == set(
        [
            "loss",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "regularization_loss",
            "recall_at_10",
            "precision_at_10",
        ]
    )

    # 7. Top-k batch predict: get dataframe with top-k scores and ids
    topk_dataset = topk_encoder.batch_predict(
        dataset=music_streaming_data,
        batch_size=32,
        output_schema=music_streaming_data.schema.select_by_name("user_id"),
    )
    assert set(topk_dataset.head().columns) == set(["user_id", "top_ids", "top_scores"])

    # 8. Save and load a top-k encoder
    with tempfile.TemporaryDirectory() as tmpdir:
        topk_encoder.save(tmpdir)
        loaded_topk_encoder = tf.keras.models.load_model(tmpdir)
    batch_output = loaded_topk_encoder(batch[0])
    assert isinstance(batch_output, tuple)
    tf.assert_equal(
        topk_encoder.topk_layer._candidates,
        loaded_topk_encoder.topk_layer._candidates,
    )
    assert not loaded_topk_encoder.topk_layer._candidates.trainable
