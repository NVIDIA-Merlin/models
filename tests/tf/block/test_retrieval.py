import os.path

import pytest

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


def test_matrix_factorization_block(music_streaming_data: SyntheticData):
    mf = ml.MatrixFactorizationBlock(music_streaming_data.schema, dim=128)

    outputs = mf(music_streaming_data.tf_tensor_dict)

    assert "user_id" in outputs
    assert "item_id" in outputs


def test_matrix_factorization_embedding_export(music_streaming_data: SyntheticData, tmp_path):
    from merlin_models.tf.block.retrieval import CosineSimilarity

    mf = ml.MatrixFactorizationBlock(
        music_streaming_data.schema, dim=128, aggregation=CosineSimilarity()
    )
    model = mf.connect(ml.BinaryClassificationTask("like"))
    model.compile(optimizer="adam")

    model.fit(music_streaming_data.tf_dataloader(), epochs=5)

    item_embedding_parquet = str(tmp_path / "items.parquet")
    mf.export_embedding_table(Tag.ITEM_ID, item_embedding_parquet, gpu=False)
    assert os.path.exists(item_embedding_parquet)

    # Test GPU export if available
    try:
        import cudf  # noqa: F401

        user_embedding_parquet = str(tmp_path / "users.parquet")
        mf.export_embedding_table(Tag.USER_ID, user_embedding_parquet, gpu=True)
        assert os.path.exists(user_embedding_parquet)
    except ImportError:
        pass
