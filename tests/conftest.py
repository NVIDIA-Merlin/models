from pathlib import Path

import pytest

NUM_EXAMPLES = 1000
CARDINALITY = 100
VECTOR_DIM = 128
N_HOT = 5

@pytest.fixture
def tmpdir():
    tmp = Path("./tmp")
    tmp.mkdir(exist_ok=True)
    return tmp


@pytest.fixture
def continuous_columns():
    tf = pytest.importorskip("tensorflow")
    return [
        tf.feature_column.numeric_column("scalar_continuous", (1,)),
        tf.feature_column.numeric_column("vector_continuous", (VECTOR_DIM,)),
    ]


@pytest.fixture
def categorical_columns():
    tf = pytest.importorskip("tensorflow")
    return [
        tf.feature_column.categorical_column_with_identity("one_hot_a", CARDINALITY),
        tf.feature_column.categorical_column_with_identity("one_hot_b", CARDINALITY),
        tf.feature_column.categorical_column_with_identity("multi_hot_a", CARDINALITY),
    ]


@pytest.fixture
def continuous_features():
    tf = pytest.importorskip("tensorflow")

    scalar_feature = tf.random.uniform((NUM_EXAMPLES, 1))
    vector_feature = tf.random.uniform((NUM_EXAMPLES, VECTOR_DIM))

    return {
        "scalar_continuous": scalar_feature,
        "vector_continuous__values": vector_feature,
    }


@pytest.fixture
def categorical_features():
    tf = pytest.importorskip("tensorflow")

    one_hot_a = tf.random.uniform((NUM_EXAMPLES, 1), maxval=CARDINALITY, dtype=tf.dtypes.int32)
    one_hot_b = tf.random.uniform((NUM_EXAMPLES, 1), maxval=CARDINALITY, dtype=tf.dtypes.int32)

    nnzs = 5
    multi_hot_a__nnzs = tf.fill((NUM_EXAMPLES, 1), nnzs)
    multi_hot_a__values = tf.random.uniform((NUM_EXAMPLES, N_HOT), maxval=CARDINALITY, dtype=tf.dtypes.int32)

    return {
        "one_hot_a": one_hot_a,
        "one_hot_b": one_hot_b,
        "multi_hot_a__nnzs": multi_hot_a__nnzs,
        "multi_hot_a__values": multi_hot_a__values,
    }


@pytest.fixture
def labels():
    tf = pytest.importorskip("tensorflow")

    labels = tf.random.uniform((NUM_EXAMPLES, 1), maxval=2, dtype=tf.dtypes.int32)

    return labels


def transform_for_inference(training_data):
    inference_data = {}

    for channel_name, channel_features in training_data.items():
        for feature_name, feature in channel_features.items():
            inference_data[f"{channel_name}_{feature_name}"] = feature

    return inference_data
