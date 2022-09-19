import merlin.models.tf as mm
from merlin.schema import ColumnSchema, Schema, Tags


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = mm.Filter(features)(tf_con_features)

    assert list(con.keys()) == features


def test_filter_call(tf_con_features):
    assert mm.Filter([])({}) == {}
    assert mm.Filter([])(tf_con_features) == {}
    assert mm.Filter(["unknown"])(tf_con_features) == {}

    assert mm.Filter(Schema())({}) == {}
    assert mm.Filter(Schema())(tf_con_features) == {}
    assert mm.Filter(Schema(["unknown"]))(tf_con_features) == {}

    assert mm.Filter(Tags.CONTINUOUS)({}) == {}
    assert mm.Filter(Tags.CONTINUOUS).set_schema(Schema(["unknown"]))(tf_con_features) == {}


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = mm.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_block(tf_con_features):
    _DummyTabular = mm.TabularBlock

    tabular = _DummyTabular()

    assert tabular(tf_con_features) == tf_con_features
    assert tabular(tf_con_features, aggregation="concat").shape[1] == 6
    assert tabular(tf_con_features, aggregation=mm.ConcatFeatures()).shape[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(tf_con_features).shape[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(tf_con_features, merge_with=tab_b, aggregation="stack").shape[1] == 1


def test_tabular_block_inputs_are_filtered(tf_con_features):

    no_schema = mm.TabularBlock()
    assert no_schema.has_schema is False

    assert no_schema(tf_con_features) == tf_con_features
    assert no_schema(tf_con_features, aggregation="concat").shape[1] == 6

    schema = Schema(
        [
            ColumnSchema("a", tags=[Tags.USER, Tags.CONTINUOUS]),
            ColumnSchema("b", tags=[Tags.ITEM, Tags.CONTINUOUS]),
            ColumnSchema("c", tags=[Tags.USER, Tags.CONTINUOUS]),
            ColumnSchema("d", tags=[Tags.ITEM, Tags.CONTINUOUS]),
            ColumnSchema("e", tags=[Tags.ITEM, Tags.CONTINUOUS]),
            ColumnSchema("f", tags=[Tags.ITEM, Tags.CONTINUOUS]),
        ]
    )
    user_features = mm.TabularBlock(schema=schema.select_by_tag(Tags.USER))
    assert user_features.schema == schema.select_by_tag(Tags.USER)
    assert user_features(tf_con_features) == {k: v for k, v in tf_con_features.items() if k in "ac"}
    assert user_features(tf_con_features, aggregation="concat").shape[1] == 2

    item_features = mm.TabularBlock(schema=schema.select_by_tag(Tags.ITEM))
    assert item_features.schema == schema.select_by_tag(Tags.ITEM)
    assert item_features(tf_con_features) == {
        k: v for k, v in tf_con_features.items() if k in "bdef"
    }
    assert item_features(tf_con_features, aggregation="concat").shape[1] == 4
