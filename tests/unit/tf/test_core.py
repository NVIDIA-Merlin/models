from typing import List

import pytest
import tensorflow as tf
from tensorflow.keras import mixed_precision

import merlin.models.tf as ml
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = ml.Filter(features)(tf_con_features)

    assert list(con.keys()) == features


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = ml.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_block(tf_con_features):
    _DummyTabular = ml.TabularBlock

    tabular = _DummyTabular()

    assert tabular(tf_con_features) == tf_con_features
    assert tabular(tf_con_features, aggregation="concat").shape[1] == 6
    assert tabular(tf_con_features, aggregation=ml.ConcatFeatures()).shape[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(tf_con_features).shape[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(tf_con_features, merge_with=tab_b, aggregation="stack").shape[1] == 1


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


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DummyFeaturesBlock(ml.Block):
    def add_features_to_context(self, feature_shapes) -> List[str]:
        return [Tags.ITEM_ID.value]

    def call(self, inputs, feature_context, **kwargs):
        items = list(feature_context.features.select_by_tag(Tags.ITEM_ID).values.values())[0]
        emb_table = self.context.get_embedding(Tags.ITEM_ID)
        item_embeddings = tf.gather(emb_table, tf.cast(items, tf.int32))
        if tf.rank(item_embeddings) == 3:
            item_embeddings = tf.squeeze(item_embeddings)

        return inputs * item_embeddings

    def compute_output_shape(self, input_shapes):
        return input_shapes

    @property
    def item_embedding_table(self):
        return self.context.get_embedding(Tags.ITEM_ID)


@pytest.mark.parametrize("run_eagerly", [True])
def test_block_context_model(ecommerce_data: Dataset, run_eagerly: bool, tmp_path):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        DummyFeaturesBlock(),
        ml.BinaryClassificationTask("click"),
    )

    copy_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert copy_model.context == copy_model.block.layers[0].context


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
