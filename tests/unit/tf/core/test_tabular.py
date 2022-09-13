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


def test_filter_select_by_tag(tf_con_features):
    a_schema = ColumnSchema("a", tags=[Tags.CONTINUOUS])
    b_schema = ColumnSchema("b", tags=[Tags.CONTINUOUS])
    c_schema = ColumnSchema("c", tags=[Tags.CATEGORICAL])
    schema = Schema([a_schema, b_schema, c_schema])

    no_filter = mm.Filter(list("abc"))
    no_filter.set_schema(schema)

    no_filter_out = no_filter(tf_con_features)
    assert sorted(no_filter_out.keys()) == ["a", "b", "c"]

    continuous = no_filter.select_by_tag(Tags.CONTINUOUS)
    continuous_out = continuous(tf_con_features)
    assert isinstance(continuous, mm.Filter)
    assert sorted(continuous_out.keys()) == ["a", "b"]


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


def test_tabular_block_select_by_tag(tf_con_features):
    tabular_block = mm.TabularBlock()

    a_schema = ColumnSchema("a", tags=[Tags.CONTINUOUS, Tags.USER])
    b_schema = ColumnSchema("b", tags=[Tags.CONTINUOUS, Tags.ITEM])
    c_schema = ColumnSchema("c", tags=[Tags.CONTINUOUS, Tags.ITEM])
    schema = Schema([a_schema, b_schema, c_schema])

    tabular_block = mm.TabularBlock(schema=schema)
    assert sorted(tabular_block.select_by_tag([Tags.ITEM, Tags.USER]).schema.column_names) == [
        "a",
        "b",
        "c",
    ]
    assert sorted(tabular_block.select_by_tag(Tags.USER).schema.column_names) == ["a"]
    assert sorted(tabular_block.select_by_tag(Tags.ITEM).schema.column_names) == ["b", "c"]
