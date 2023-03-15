import tempfile

import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.transforms.features import expected_input_cols_from_schema
from merlin.models.tf.utils import testing_utils
from merlin.models.utils.dataset import unique_by_tag
from merlin.schema import Tags


def test_encoder_block(music_streaming_data: Dataset):
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "user_genres"]
    )

    schema = music_streaming_data.schema
    user_schema = schema.select_by_name(["user_id", "user_genres"])
    user_encoder = mm.Encoder(user_schema, mm.MLPBlock([4]), name="query")
    item_schema = schema.select_by_name(["item_id"])
    item_encoder = mm.Encoder(item_schema, mm.MLPBlock([4]), name="candidate")

    model = mm.Model(
        mm.ParallelBlock(user_encoder, item_encoder),
        mm.ContrastiveOutput(item_schema, "in-batch"),
        prep_features=False,
    )

    assert model.blocks[0]["query"] == user_encoder
    assert model.blocks[0]["candidate"] == item_encoder

    loader = mm.Loader(music_streaming_data, batch_size=50)
    testing_utils.model_test(model, loader, reload_model=True)

    with pytest.raises(Exception) as excinfo:
        user_encoder.compile("adam")
        user_encoder.fit(loader)

    assert "This block is not meant to be trained by itself" in str(excinfo.value)

    user_features = expected_input_cols_from_schema(user_schema)
    testing_utils.test_model_signature(user_encoder, user_features, ["output_1"])

    item_features = expected_input_cols_from_schema(item_schema)
    testing_utils.test_model_signature(item_encoder, item_features, ["output_1"])


def test_topk_encoder(music_streaming_data: Dataset):
    TOP_K = 10
    BATCH_SIZE = 32
    music_streaming_data.schema = music_streaming_data.schema.select_by_name(
        ["user_id", "item_id", "country", "user_age"]
    )

    # 1. Train a retrieval model
    schema = music_streaming_data.schema
    user_schema = schema.select_by_name(["user_id", "country", "user_age"])
    user_encoder = mm.Encoder(user_schema, mm.MLPBlock([4]), name="query")

    item_schema = schema.select_by_name(["item_id"])
    item_encoder = mm.Encoder(item_schema, mm.MLPBlock([4]), name="candidate")

    retrieval_model = mm.RetrievalModelV2(
        query=user_encoder,
        candidate=item_encoder,
        output=mm.ContrastiveOutput(item_schema, "in-batch"),
    )
    testing_utils.model_test(retrieval_model, music_streaming_data, reload_model=True)

    # 2. Get candidates embeddings for the top-k encoder
    candidate_features = unique_by_tag(music_streaming_data, Tags.ITEM, Tags.ITEM_ID)
    candidates = retrieval_model.candidate_embeddings(
        candidate_features, batch_size=BATCH_SIZE, index=Tags.ITEM_ID
    )

    # 3. Set data-loader for top-k recommendation
    loader = mm.Loader(music_streaming_data, batch_size=BATCH_SIZE).map(
        mm.ToTarget(schema, "item_id")
    )
    batch = loader.peek()

    # 4. Define the top-k encoder
    topk_encoder = mm.TopKEncoder(
        query_encoder=retrieval_model.query_encoder, candidates=candidates, k=TOP_K
    )
    # 5. Get top-k predictions
    batch_output = topk_encoder(batch[0])
    predict_output = topk_encoder.predict(loader)

    assert list(batch_output.scores.shape) == [BATCH_SIZE, TOP_K]
    assert list(predict_output.scores.shape) == [100, TOP_K]

    # 6. Compute top-k evaluation metrics (using the whole candidates catalog)
    topk_encoder.compile()
    topk_evaluation_metrics = topk_encoder.evaluate(loader, return_dict=True)
    assert set(topk_evaluation_metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "regularization_loss",
            "recall_at_10",
            "precision_at_10",
        ]
    )

    # 7. Top-k batch predict: get a dataframe with top-k scores and ids
    topk_dataset = topk_encoder.batch_predict(
        dataset=music_streaming_data,
        batch_size=32,
        output_schema=music_streaming_data.schema.select_by_name("user_id"),
    ).compute()
    assert len(topk_dataset.head().columns) == 1 + (TOP_K * 2)

    # 8. Save and load the top-k encoder
    with tempfile.TemporaryDirectory() as tmpdir:
        topk_encoder.save(tmpdir)
        loaded_topk_encoder = tf.keras.models.load_model(tmpdir)
    batch_output = loaded_topk_encoder(batch[0])

    assert list(batch_output.scores.shape) == [BATCH_SIZE, TOP_K]
    tf.debugging.assert_equal(
        topk_encoder.topk_layer._candidates,
        loaded_topk_encoder.topk_layer._candidates,
    )

    assert not loaded_topk_encoder.topk_layer._candidates.trainable

    # 9. Change the top-k threshold
    scores = topk_encoder(batch[0], k=20)
    assert list(scores.scores.shape) == [BATCH_SIZE, 20]
    scores = topk_encoder(batch[0], k=30)
    assert list(scores.scores.shape) == [BATCH_SIZE, 30]

    topk_encoder.compile(k=20)
    scores = topk_encoder.predict(loader)
    assert list(scores.scores.shape) == [100, 20]
    topk_encoder.compile(k=30)
    scores = topk_encoder.predict(loader)
    assert list(scores.scores.shape) == [100, 30]
