import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.predictions.dot_product import (
    ContrastiveDotProduct,
    DotProductCategoricalPrediction,
)
from merlin.models.tf.predictions.sampling.in_batch import InBatchSamplerV2
from merlin.models.tf.utils import testing_utils


def test_dot_product_prediction(ecommerce_data: Dataset):
    model = mm.RetrievalModel(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        DotProductCategoricalPrediction(
            ecommerce_data.schema, negative_samplers=InBatchSamplerV2()
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data)


@testing_utils.mark_run_eagerly_modes
def test_setting_negative_sampling_strategy(ecommerce_data: Dataset, run_eagerly: bool):
    model = mm.RetrievalModel(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        DotProductCategoricalPrediction(ecommerce_data.schema),
    )
    model.compile(run_eagerly=run_eagerly, optimizer="adam", negative_sampling="in-batch")
    _ = model.fit(ecommerce_data, batch_size=50, epochs=5, steps_per_epoch=1)


def test_add_sampler(ecommerce_data: Dataset):
    model = mm.RetrievalModel(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        DotProductCategoricalPrediction(
            ecommerce_data.schema, negative_samplers=InBatchSamplerV2()
        ),
    )
    assert len(model.prediction_blocks[0].prediction_with_negatives.negative_samplers) == 1
    model.prediction_blocks[0].add_sampler(mm.InBatchSamplerV2())
    assert len(model.prediction_blocks[0].prediction_with_negatives.negative_samplers) == 2


def test_contrastive_dot_product(ecommerce_data: Dataset):
    batch_size = 10
    inbatch_sampler = InBatchSamplerV2()

    retrieval_scorer = ContrastiveDotProduct(
        schema=ecommerce_data.schema,
        negative_samplers=[inbatch_sampler],
        downscore_false_negatives=False,
    )
    inputs, features = _retrieval_inputs_(batch_size=batch_size)
    output = retrieval_scorer(inputs, features=features)

    expected_num_samples_inbatch = batch_size + 1
    tf.assert_equal(tf.shape(output.predictions)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(tf.shape(output.predictions)[1], expected_num_samples_inbatch)


def test_item_retrieval_scorer_no_sampler(ecommerce_data: Dataset):
    with pytest.raises(Exception) as excinfo:
        inputs, features = _retrieval_inputs_(batch_size=10)
        retrieval_scorer = ContrastiveDotProduct(
            schema=ecommerce_data.schema, negative_samplers=[], downscore_false_negatives=False
        )
        _ = retrieval_scorer(inputs, features=features, training=True)
    assert "At least one sampler is required by ContrastiveDotProduct for negative sampling" in str(
        excinfo.value
    )


def test_item_retrieval_scorer_downscore_false_negatives(ecommerce_data: Dataset):
    batch_size = 10

    inbatch_sampler = InBatchSamplerV2()
    inputs, features = _retrieval_inputs_(batch_size=batch_size)

    FALSE_NEGATIVE_SCORE = -100_000_000.0
    item_retrieval_scorer = ContrastiveDotProduct(
        schema=ecommerce_data.schema,
        negative_samplers=[inbatch_sampler],
        downscore_false_negatives=True,
        false_negative_score=FALSE_NEGATIVE_SCORE,
    )

    outputs = item_retrieval_scorer(
        inputs,
        training=True,
        features=features,
    )
    output_scores = outputs.predictions

    output_neg_scores = output_scores[:, 1:]

    diag_mask = tf.eye(tf.shape(output_neg_scores)[0], dtype=tf.bool)
    tf.assert_equal(output_neg_scores[diag_mask], FALSE_NEGATIVE_SCORE)
    tf.assert_equal(
        tf.reduce_all(
            tf.not_equal(
                output_neg_scores[tf.math.logical_not(diag_mask)],
                tf.constant(FALSE_NEGATIVE_SCORE, dtype=output_neg_scores.dtype),
            )
        ),
        True,
    )


def test_retrieval_prediction_only_positive_when_not_training(ecommerce_data: Dataset):
    batch_size = 10

    inbatch_sampler = InBatchSamplerV2()
    item_retrieval_prediction = DotProductCategoricalPrediction(
        schema=ecommerce_data.schema,
        negative_samplers=[inbatch_sampler],
        downscore_false_negatives=False,
    )

    inputs, features = _retrieval_inputs_(batch_size=batch_size)
    output_scores = item_retrieval_prediction(inputs)
    tf.assert_equal(
        (int(tf.shape(output_scores)[0]), int(tf.shape(output_scores)[1])), (batch_size, 1)
    )


def _retrieval_inputs_(batch_size):
    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    positive_items = tf.random.uniform(
        shape=(batch_size,), minval=1, maxval=1000000, dtype=tf.int32
    )
    inputs = {"query": users_embeddings, "item": items_embeddings, "item_id": positive_items}
    features = {"product_id": positive_items, "user_id": None}
    return inputs, features
