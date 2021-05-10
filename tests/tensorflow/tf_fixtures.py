from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")
models = pytest.importorskip("merlin_models.tensorflow.models")


@pytest.fixture
def tmpdir():
    tmp = Path("./tmp")
    tmp.mkdir(exist_ok=True)
    return tmp


@pytest.fixture
def continuous_columns():
    return [
        tf.feature_column.numeric_column("scalar_continuous", (1,)),
        tf.feature_column.numeric_column("vector_continuous", (128,)),
    ]


@pytest.fixture
def categorical_columns():
    return [
        tf.feature_column.categorical_column_with_identity("one_hot_a", 100),
        tf.feature_column.categorical_column_with_identity("one_hot_b", 100),
    ]


@pytest.fixture
def continuous_features():
    scalar_feature = tf.random.uniform((1000, 1))
    vector_feature = tf.random.uniform((1000, 128))

    return {
        "scalar_continuous": scalar_feature,
        "vector_continuous__values": vector_feature,
    }


@pytest.fixture
def categorical_features():
    one_hot_a = tf.random.uniform((1000, 1), maxval=100, dtype=tf.dtypes.int32)
    one_hot_b = tf.random.uniform((1000, 1), maxval=100, dtype=tf.dtypes.int32)

    return {
        "one_hot_a": one_hot_a,
        "one_hot_b": one_hot_b,
    }


@pytest.fixture
def labels():
    labels = tf.random.uniform((1000, 1), maxval=1, dtype=tf.dtypes.int32)

    return labels


def transform_for_inference(training_data):
    inference_data = {}

    for channel_name, channel_features in training_data.items():
        for feature_name, feature in channel_features.items():
            inference_data[f"{channel_name}_{feature_name}"] = feature

    return inference_data
