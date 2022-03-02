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

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData
from merlin.schema import Tags


def test_matrix_factorization_block(music_streaming_data: SyntheticData):
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128)

    outputs = mf(music_streaming_data.tf_tensor_dict)

    assert "query" in outputs
    assert "item" in outputs


def test_matrix_factorization_embedding_export(music_streaming_data: SyntheticData, tmp_path):
    import pandas as pd

    from merlin.models.tf.blocks.core.aggregation import CosineSimilarity

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
