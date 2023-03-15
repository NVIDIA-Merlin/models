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
import os
import time
import timeit

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import merlin.models.tf as mm
from merlin.core.dispatch import make_df
from merlin.io.dataset import Dataset
from merlin.models.tf.utils.tf_utils import list_col_to_ragged
from merlin.models.utils.schema_utils import create_categorical_column
from merlin.schema import ColumnSchema, Schema, Tags


def test_lazy_dataset_map():
    dataset_size = 100
    data_df = pd.DataFrame({"feature": np.random.randint(100, size=dataset_size)})
    schema = Schema([ColumnSchema("feature", tags=[Tags.CATEGORICAL])])
    dataset = Dataset(data_df, schema=schema)

    sleep_time_seconds = 0.5

    def identity(x, y):
        time.sleep(sleep_time_seconds)
        return (x, y)

    loader = mm.Loader(dataset, batch_size=10).map(identity)

    elapsed_time_seconds = timeit.timeit(lambda: next(iter(loader)), number=1)

    assert elapsed_time_seconds < 1


def test_nested_list():
    num_rows = 100
    batch_size = 12

    df = pd.DataFrame(
        {
            "data": [
                np.random.rand(np.random.randint(10) + 1, 3).tolist() for i in range(num_rows)
            ],
            "data2": [np.random.rand(np.random.randint(10) + 1).tolist() for i in range(num_rows)],
            "label": [np.random.rand() for i in range(num_rows)],
        }
    )

    loader = mm.Loader(
        Dataset(df),
        cont_names=["data", "data2"],
        label_names=["label"],
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(loader))

    @tf.function
    def _ragged_to_dense(col_name):
        result = list_col_to_ragged(
            batch[0][f"{col_name}__values"], batch[0][f"{col_name}__offsets"]
        ).to_tensor()
        return result

    # [[1,2,3],[3,1],[...],[]]
    nested_data_col = _ragged_to_dense("data")
    true_data_col = tf.reshape(
        tf.ragged.constant(df.iloc[:batch_size, 0].tolist()).to_tensor(), [batch_size, -1]
    )

    # [1,2,3]
    multihot_data2_col = _ragged_to_dense("data2")
    true_data2_col = tf.reshape(
        tf.ragged.constant(df.iloc[:batch_size, 1].tolist()).to_tensor(), [batch_size, -1]
    )
    assert nested_data_col.shape == true_data_col.shape
    assert np.allclose(nested_data_col.numpy(), true_data_col.numpy())
    assert multihot_data2_col.shape == true_data2_col.shape
    assert np.allclose(multihot_data2_col.numpy(), true_data2_col.numpy())


def test_shuffling():
    num_rows = 10000
    batch_size = 10000

    df = pd.DataFrame({"a": np.asarray(range(num_rows)), "b": np.asarray([0] * num_rows)})

    loader = mm.Loader(
        Dataset(df), cont_names=["a"], label_names=["b"], batch_size=batch_size, shuffle=True
    )

    batch = next(iter(loader))

    first_batch = tf.reshape(tf.cast(batch[0]["a"].cpu(), tf.int32), (batch_size,))
    in_order = tf.range(0, batch_size, dtype=tf.int32)

    assert (first_batch != in_order).numpy().any()
    assert (tf.sort(first_batch) == in_order).numpy().all()


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_tf_drp_reset(tmpdir, batch_size, drop_last, num_rows):
    df = make_df(
        {
            "cat1": [1] * num_rows,
            "cat2": [2] * num_rows,
            "cat3": [3] * num_rows,
            "label": [0] * num_rows,
            "cont3": [3.0] * num_rows,
            "cont2": [2.0] * num_rows,
            "cont1": [1.0] * num_rows,
        }
    )
    path = os.path.join(tmpdir, "Dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["cont3", "cont2", "cont1"]
    label_name = ["label"]

    loader = mm.Loader(
        path,
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        label_names=label_name,
        shuffle=False,
        drop_last=drop_last,
    )

    all_len = len(loader) if drop_last else len(loader) - 1
    all_rows = 0
    for idx, (X, y) in enumerate(loader):
        all_rows += len(X["cat1"])
        if idx < all_len:
            assert list(X["cat1"].numpy()) == [1] * batch_size
            assert list(X["cat2"].numpy()) == [2] * batch_size
            assert list(X["cat3"].numpy()) == [3] * batch_size
            assert list(X["cont1"].numpy()) == [1.0] * batch_size
            assert list(X["cont2"].numpy()) == [2.0] * batch_size
            assert list(X["cont3"].numpy()) == [3.0] * batch_size

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


def test_tf_catname_ordering(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "cont3": [3.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "Dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["cont3", "cont2", "cont1"]
    label_name = ["label"]

    loader = mm.Loader(
        path,
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=10,
        label_names=label_name,
        shuffle=False,
    )

    for X, y in loader:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10
        assert list(X["cont3"].numpy()) == [3.0] * 10


def test_tf_map(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "sample_weight": [1.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "Dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["sample_weight", "cont2", "cont1"]
    label_name = ["label"]

    def add_sample_weight(features, labels, sample_weight_col_name="sample_weight"):
        sample_weight = tf.cast(features.pop(sample_weight_col_name) > 0, tf.float32)

        return features, labels, sample_weight

    loader = mm.Loader(
        path,
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=10,
        label_names=label_name,
        shuffle=False,
    ).map(add_sample_weight)

    for X, y, sample_weight in loader:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10

        assert list(sample_weight.numpy()) == [1.0] * 10


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_validator(batch_size):
    n_samples = 9
    rand = np.random.RandomState(0)

    gdf = make_df({"a": rand.randn(n_samples), "label": rand.randint(2, size=n_samples)})

    loader = mm.Loader(
        Dataset(gdf),
        batch_size=batch_size,
        cat_names=[],
        cont_names=["a"],
        label_names=["label"],
        shuffle=False,
    )

    input_ = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    x = tf.keras.layers.Dense(128, "relu")(input_)
    x = tf.keras.layers.Dense(1, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_, outputs=x)
    model.compile("sgd", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

    validator = mm.KerasSequenceValidator(loader)
    model.fit(loader, epochs=2, verbose=0, callbacks=[validator])

    predictions, labels = [], []
    for X, y_true in loader:
        y_pred = model(X)
        labels.extend(y_true.numpy())
        predictions.extend(y_pred.numpy()[:, 0])
    predictions = np.array(predictions)
    labels = np.array(labels)

    logs = {}
    validator.on_epoch_end(0, logs)
    auc_key = [i for i in logs if i.startswith("val_auc")][0]

    true_accuracy = (labels == (predictions > 0.5)).mean()
    estimated_accuracy = logs["val_accuracy"]
    assert np.isclose(true_accuracy, estimated_accuracy, rtol=1e-1)

    true_auc = roc_auc_score(labels, predictions)
    estimated_auc = logs[auc_key]
    assert np.isclose(true_auc, estimated_auc, rtol=1e-1)


def test_block_with_sparse_inputs(music_streaming_data: Dataset):
    item_id_schema = music_streaming_data.schema.select_by_name(["user_id", "item_genres"])

    inputs = mm.InputBlock(item_id_schema)
    block = inputs.connect(mm.MLPBlock([64]), context=mm.ModelContext())

    df = pd.DataFrame(
        {
            "item_genres": np.random.randint(0, 10, (32, 20)).tolist(),
            "user_id": np.random.randint(0, 10, (32,)).tolist(),
        }
    )
    loader = mm.Loader(
        Dataset(df),
        cat_names=["user_id", "item_genres"],
        batch_size=3,
        shuffle=False,
    )

    batch = next(iter(loader))[0]
    out = block(batch)
    assert out.shape[-1] == 64


def test_block_with_categorical_target():
    import pandas as pd

    from merlin.schema import Schema, Tags

    df = pd.DataFrame(
        {
            "Author": [12, 4, 23, 19],
            "Engaging User": [23, 23, 12, 5],
            "target": [1, 2, 3, 4],
        }
    )
    s = Schema(
        [
            create_categorical_column("Engaging User", num_items=24, tags=[Tags.CATEGORICAL]),
            create_categorical_column("Author", num_items=24, tags=[Tags.CATEGORICAL]),
            create_categorical_column("target", num_items=5, tags=[Tags.CATEGORICAL, Tags.TARGET]),
        ]
    )
    data = Dataset(df, schema=s)

    batch = mm.sample_batch(data, batch_size=2)
    assert batch[1].shape == (2, 1)

    inputs = mm.InputBlock(data.schema)
    embeddings = inputs(batch[0])
    assert list(embeddings.keys()) == ["Engaging User", "Author"]
