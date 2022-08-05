import pytest
from tensorflow.keras import mixed_precision

import merlin.models.tf as ml
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("pre", [None])
@pytest.mark.parametrize("post", [None])
@pytest.mark.parametrize("aggregation", [None, "concat"])
@pytest.mark.parametrize("include_schema", [True, False])
def test_serialization_continuous_features(
    testing_data: Dataset, pre, post, aggregation, include_schema
):
    schema = None
    if include_schema:
        schema = testing_data.schema

    inputs = ml.TabularBlock(pre=pre, post=post, aggregation=aggregation, schema=schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    keep_cols = ["user_id", "item_id", "event_hour_sin", "event_hour_cos"]
    tf_tabular_data = ml.sample_batch(testing_data, batch_size=100, include_targets=False)
    for k in list(tf_tabular_data.keys()):
        if k not in keep_cols:
            del tf_tabular_data[k]

    assert copy_layer(tf_tabular_data) is not None
    assert inputs.pre.__class__.__name__ == copy_layer.pre.__class__.__name__
    assert inputs.post.__class__.__name__ == copy_layer.post.__class__.__name__
    assert inputs.aggregation.__class__.__name__ == copy_layer.aggregation.__class__.__name__
    if include_schema:
        assert inputs.schema == schema


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_block_context_model_fp16(ecommerce_data: Dataset, run_eagerly: bool, num_epochs=2):
    mixed_precision.set_global_policy("mixed_float16")
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([32]),
        ml.BinaryClassificationTask("click"),
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    mixed_precision.set_global_policy("float32")
    losses = model.fit(ecommerce_data, batch_size=100, epochs=2)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])
