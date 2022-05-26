#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.schema import Tags


def test_topk_index(ecommerce_data: Dataset):
    import tensorflow as tf

    from merlin.models.tf.metrics.evaluation import ItemCoverageAt, PopularityBiasAt

    model: mm.RetrievalModel = mm.TwoTowerModel(
        ecommerce_data.schema, query_tower=mm.MLPBlock([64, 128])
    )
    model.compile(run_eagerly=False, optimizer="adam")
    model.fit(ecommerce_data, epochs=1, batch_size=50)

    item_features = ecommerce_data.schema.select_by_tag(Tags.ITEM).column_names
    item_dataset = ecommerce_data.to_ddf()[item_features].drop_duplicates().compute()
    item_dataset = Dataset(item_dataset)
    recommender = model.to_top_k_recommender(item_dataset, k=20)
    NUM_ITEMS = 1001
    item_frequency = tf.sort(
        tf.random.uniform((NUM_ITEMS,), minval=0, maxval=NUM_ITEMS, dtype=tf.int32)
    )
    eval_metrics = [
        PopularityBiasAt(item_freq_probs=item_frequency, is_prob_distribution=False, k=10),
        ItemCoverageAt(num_unique_items=NUM_ITEMS, k=10),
    ]
    batch = mm.sample_batch(ecommerce_data, batch_size=10, include_targets=False)
    _, top_indices = recommender(batch)
    assert top_indices.shape[-1] == 20
    _, top_indices = recommender(batch, k=10)
    assert top_indices.shape[-1] == 10

    for metric in eval_metrics:
        metric.update_state(predicted_ids=top_indices)
    for metric in eval_metrics:
        results = metric.result()
        metric.reset_state()
        assert results >= 0


def test_topk_index_duplicate_indices(ecommerce_data: Dataset):
    model: mm.RetrievalModel = mm.TwoTowerModel(
        ecommerce_data.schema, query_tower=mm.MLPBlock([64, 128])
    )
    model.compile(run_eagerly=True, optimizer="adam")
    model.fit(ecommerce_data, epochs=1, batch_size=50)
    item_features = ecommerce_data.schema.select_by_tag(Tags.ITEM).column_names
    item_dataset = ecommerce_data.to_ddf()[item_features].compute()
    item_dataset = Dataset(item_dataset)

    with pytest.raises(ValueError) as excinfo:
        _ = model.to_top_k_recommender(item_dataset, k=20)
    assert "Please make sure that `data` contains unique indices" in str(excinfo.value)


def test_topk_recommender_outputs(ecommerce_data: Dataset, batch_size=100):
    import numpy as np
    import tensorflow as tf

    import merlin.models.tf.dataset as tf_dataloader
    from merlin.models.tf.blocks.core.index import IndexBlock
    from merlin.models.utils.dataset import unique_rows_by_features

    def numpy_recall(labels, top_item_ids, k):
        return np.equal(np.expand_dims(labels, -1), top_item_ids[:, :k]).max(axis=-1).mean()

    model = mm.TwoTowerModel(
        ecommerce_data.schema,
        query_tower=mm.MLPBlock([64]),
        samplers=[mm.InBatchSampler()],
    )

    model.compile("adam", metrics=[mm.RecallAt(10)])
    model.fit(ecommerce_data, batch_size=batch_size, epochs=3)
    eval_metrics = model.evaluate(
        ecommerce_data, item_corpus=ecommerce_data, batch_size=batch_size, return_dict=True
    )

    # Manually compute top-k ids for a given batch
    batch = mm.sample_batch(ecommerce_data, batch_size=batch_size, include_targets=False)
    item_dataset = unique_rows_by_features(ecommerce_data, Tags.ITEM, Tags.ITEM_ID)
    candidates_dataset_df = IndexBlock.get_candidates_dataset(
        block=model.retrieval_block.item_block(), data=item_dataset, id_column="item_id"
    )
    item_tower_ids, item_tower_embeddings = IndexBlock.extract_ids_embeddings(
        candidates_dataset_df, check_unique_ids=True
    )
    batch_query_tower_embeddings = model.retrieval_block.query_block()(batch)
    batch_user_scores_all_items = tf.matmul(
        batch_query_tower_embeddings, item_tower_embeddings, transpose_b=True
    )
    top_scores, top_indices = tf.math.top_k(batch_user_scores_all_items, k=10)
    top_ids = tf.gather(item_tower_ids, top_indices)

    # Get top-k ids from the topk_recommender_model
    topk_recommender_model = model.to_top_k_recommender(ecommerce_data, k=10)
    topk_predictions, topk_items = topk_recommender_model(batch)

    # Assert top-k items from top-k recommender are the same as the manually computed top-k items
    tf.debugging.assert_equal(top_ids, topk_items)

    # Compute recall using top-k recommender
    data_dl = tf_dataloader.BatchedDataset(
        ecommerce_data,
        batch_size=batch_size,
        shuffle=False,
    )
    topk_output = topk_recommender_model.predict(data_dl)
    topk_predictions, topk_items = topk_output
    test_df = ecommerce_data.to_ddf()
    positive_item_ids = np.array(test_df["item_id"].compute().values.tolist())
    recall_at_10 = numpy_recall(positive_item_ids, topk_items, k=10)

    np.isclose(recall_at_10, eval_metrics["recall_at_10"], rtol=1e-6)
