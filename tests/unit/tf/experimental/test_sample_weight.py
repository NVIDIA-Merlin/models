import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.experimental.sample_weight import ContrastiveSampleWeight
from merlin.models.tf.metrics.topk import NDCGAt, RecallAt
from merlin.models.tf.outputs.base import DotProduct
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


@pytest.mark.parametrize(
    "pos_class_weight", [0.9, tf.random.uniform(shape=(1000,)), "user_item_intentions"]
)
@pytest.mark.parametrize("neg_class_weight", [0.1, tf.random.uniform(shape=(1000,))])
def test_two_tower_v2_with_contrastive_sample_weight(
    ecommerce_data: Dataset,
    pos_class_weight,
    neg_class_weight,
):
    tower_dim = 64
    data = ecommerce_data
    input_features = ["user_categories", "item_id"]
    if isinstance(pos_class_weight, str):
        input_features += [pos_class_weight]
    data.schema = data.schema.select_by_name(input_features)

    user_schema = data.schema.select_by_tag(Tags.USER)
    user_inputs = mm.InputBlockV2(user_schema)
    query = mm.Encoder(user_inputs, mm.MLPBlock([128, tower_dim]))

    item_schema = data.schema.select_by_tag(Tags.ITEM)
    item_inputs = mm.InputBlockV2(item_schema)
    candidate = mm.Encoder(item_inputs, mm.MLPBlock([128, tower_dim]))

    output = mm.ContrastiveOutput(
        DotProduct(),
        post=None,
        schema=data.schema.select_by_tag(Tags.ITEM_ID),
        negative_samplers="in-batch",
        store_negative_ids=True,
    )
    model = mm.TwoTowerModelV2(query, candidate, outputs=output, schema=data.schema)

    weighted_metrics = [RecallAt(10), NDCGAt(10)]

    model.compile(
        run_eagerly=True,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[RecallAt(10), NDCGAt(10)],
        weighted_metrics=weighted_metrics,
    )

    batch = mm.sample_batch(data, batch_size=16)

    # metrics without sample weights
    metrics1 = model.test_step(batch)
    assert metrics1["loss"] >= 0
    metrics2 = model.test_step(batch)
    assert metrics1["loss"] == metrics2["loss"]
    for m in metrics1:
        assert metrics2[m] == metrics2[m]

    # metrics with 2-D sample weights
    model.blocks[-1].post = ContrastiveSampleWeight(
        pos_class_weight=pos_class_weight, neg_class_weight=neg_class_weight, schema=data.schema
    )
    model.compiled_metrics.reset_state()
    metrics3 = model.test_step(batch)

    assert metrics3["loss"] != metrics2["loss"]
    for m in metrics3:
        if m.startswith("weighted_"):
            assert metrics3[m] <= 1
            if isinstance(pos_class_weight, tf.Tensor):
                assert metrics3[m] != metrics2[m]


@testing_utils.mark_run_eagerly_modes
def test_contrastive_sample_weight_serialization(ecommerce_data: Dataset, run_eagerly):
    tower_dim = 64
    data = ecommerce_data
    input_features = ["user_categories", "item_id"]
    data.schema = data.schema.select_by_name(input_features)

    user_schema = data.schema.select_by_tag(Tags.USER)
    user_inputs = mm.InputBlockV2(user_schema)
    query = mm.Encoder(user_inputs, mm.MLPBlock([128, tower_dim]))

    item_schema = data.schema.select_by_tag(Tags.ITEM)
    item_inputs = mm.InputBlockV2(item_schema)
    candidate = mm.Encoder(item_inputs, mm.MLPBlock([128, tower_dim]))

    item_id_cardinality = item_schema["item_id"].int_domain.max + 1

    output = mm.ContrastiveOutput(
        DotProduct(),
        post=ContrastiveSampleWeight(
            pos_class_weight=tf.random.uniform(shape=(item_id_cardinality,)),
            neg_class_weight=tf.random.uniform(shape=(item_id_cardinality,)),
            schema=data.schema,
        ),
        schema=data.schema.select_by_tag(Tags.ITEM_ID),
        negative_samplers="in-batch",
        store_negative_ids=True,
    )
    model = mm.TwoTowerModelV2(query, candidate, outputs=output, schema=data.schema)

    # test the model's serialization
    _ = testing_utils.model_test(model, data, reload_model=True, run_eagerly=run_eagerly)
