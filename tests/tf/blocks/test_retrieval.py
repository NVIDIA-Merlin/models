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
import os.path

import pytest
from merlin.schema import Tags

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData
from merlin_models.tf.blocks.retrieval import ElementWiseMultiply
from merlin_models.tf.utils import testing_utils


def test_matrix_factorization_block(music_streaming_data: SyntheticData):
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128)

    outputs = mf(music_streaming_data.tf_tensor_dict)

    assert "query" in outputs
    assert "item" in outputs


def test_matrix_factorization_embedding_export(music_streaming_data: SyntheticData, tmp_path):
    import pandas as pd

    from merlin_models.tf.blocks.retrieval import CosineSimilarity

    mf = ml.MatrixFactorizationBlock(
        music_streaming_data.schema, dim=128, aggregation=CosineSimilarity()
    )
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128, aggregation="cosine")
    model = mf.connect(ml.BinaryClassificationTask("like"))
    model.compile(optimizer="adam")

    model.fit(music_streaming_data.tf_dataloader(), epochs=5)

    item_embedding_parquet = str(tmp_path / "items.parquet")
    mf.export_embedding_table(Tags.ITEM_ID, item_embedding_parquet, gpu=False)

    df = mf.embedding_table_df(Tags.ITEM_ID, gpu=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10001
    assert os.path.exists(item_embedding_parquet)

    # Test GPU export if available
    try:
        import cudf  # noqa: F401

        user_embedding_parquet = str(tmp_path / "users.parquet")
        mf.export_embedding_table(Tags.USER_ID, user_embedding_parquet, gpu=True)
        assert os.path.exists(user_embedding_parquet)
        df = mf.embedding_table_df(Tags.USER_ID, gpu=True)
        assert isinstance(df, cudf.DataFrame)
        assert len(df) == 10001
    except ImportError:
        pass



def test_elementwisemultiply():
    emb1 = np.random.uniform(-1, 1, size=(5, 10))
    emb2 = np.random.uniform(-1, 1, size=(5, 10))
    x = ElementWiseMultiply()({"emb1": tf.constant(emb1), "emb2": tf.constant(emb2)})

    assert np.mean(np.isclose(x.numpy(), np.multiply(emb1, emb2))) == 1
    assert x.numpy().shape == (5, 10)


def test_two_tower_block(testing_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(testing_data.schema, query_tower=ml.MLPBlock([64, 128]))
    outputs = two_tower(testing_data.tf_tensor_dict)

    assert len(outputs) == 2
    for key in ["item", "query"]:
        assert list(outputs[key].shape) == [100, 128]


def test_two_tower_block_serialization(testing_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(testing_data.schema, query_tower=ml.MLPBlock([64, 128]))
    copy_two_tower = testing_utils.assert_serialization(two_tower)

    outputs = copy_two_tower(testing_data.tf_tensor_dict)

    assert len(outputs) == 2
    for key in ["item", "query"]:
        assert list(outputs[key].shape) == [100, 128]


def test_two_tower_block_no_item_features(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        schema = testing_data.schema.remove_by_tag(Tags.ITEM)
        ml.TwoTowerBlock(schema, query_tower=ml.MLPBlock([64]))
        assert "The schema should contain features with the tag `item`" in str(excinfo.value)


def test_two_tower_block_no_user_features(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        schema = testing_data.schema.remove_by_tag(Tags.USER)
        ml.TwoTowerBlock(schema, query_tower=ml.MLPBlock([64]))
        assert "The schema should contain features with the tag `user`" in str(excinfo.value)


def test_two_tower_block_no_schema():
    with pytest.raises(ValueError) as excinfo:
        ml.TwoTowerBlock(schema=None, query_tower=ml.MLPBlock([64]))
    assert "The schema is required by TwoTower" in str(excinfo.value)


def test_two_tower_block_no_bottom_block(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        ml.TwoTowerBlock(schema=testing_data.schema, query_tower=None)
    assert "The query_tower is required by TwoTower" in str(excinfo.value)
