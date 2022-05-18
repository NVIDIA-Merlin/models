import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/05-Retrieval-Model.ipynb", execute=False)
def test_example_05_retrieval_models(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    tb.execute()
    assert os.path.isdir("query_tower")
    assert os.path.exists("user_features.parquet")
    assert os.path.exists("item_embeddings.parquet")
    assert os.path.exists("item_features.parquet")
