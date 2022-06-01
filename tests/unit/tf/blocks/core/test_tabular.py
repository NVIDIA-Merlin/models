import merlin.models.tf as ml


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
