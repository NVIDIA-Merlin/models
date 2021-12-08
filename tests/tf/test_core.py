import pytest

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = ml.Filter(features)(tf_con_features)

    assert list(con.keys()) == features


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = ml.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_module(tf_con_features):
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
    testing_data: SyntheticData, pre, post, aggregation, include_schema
):
    from merlin_models.tf.utils.testing_utils import assert_serialization

    schema = None
    if include_schema:
        schema = testing_data.schema

    inputs = ml.TabularBlock(pre=pre, post=post, aggregation=aggregation, schema=schema)

    copy_layer = assert_serialization(inputs)

    keep_cols = ["user_id", "item_id", "event_hour_sin", "event_hour_cos"]
    tf_tabular_data = testing_data.tf_tensor_dict
    for k in list(tf_tabular_data.keys()):
        if k not in keep_cols:
            del tf_tabular_data[k]

    assert copy_layer(tf_tabular_data) is not None
    assert inputs.pre.__class__.__name__ == copy_layer.pre.__class__.__name__
    assert inputs.post.__class__.__name__ == copy_layer.post.__class__.__name__
    assert inputs.aggregation.__class__.__name__ == copy_layer.aggregation.__class__.__name__
    assert inputs.schema == schema


class DummyFeaturesBlock(ml.Block):
    def call_features(self, features, **kwargs):
        self.items = features[str(Tag.ITEM_ID)]

    def call(self, inputs, **kwargs):
        self.item_embedding_table = self.context.get_embedding(Tag.ITEM_ID)
        item_embeddings = tf.gather(self.item_embedding_table, tf.cast(self.items, tf.int32))

        return inputs * item_embeddings

    def compute_output_shape(self, input_shape):
        return input_shape


def test_block_context(ecommerce_data: SyntheticData):
    inputs = ml.inputs(ecommerce_data.schema)
    dummy = DummyFeaturesBlock()
    model = inputs.connect(ml.MLPBlock([64]), dummy)

    out = model(ecommerce_data.tf_tensor_dict)

    embeddings = inputs.select_by_name(str(Tag.CATEGORICAL))
    assert dummy.item_embedding_table.shape == embeddings.embedding_tables[str(Tag.ITEM_ID)].shape

    assert out.shape[-1] == 64
