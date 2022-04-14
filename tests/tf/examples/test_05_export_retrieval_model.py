import os

import pytest
from testbook import testbook

pytestmark = pytest.mark.example


@testbook("examples/05-Retrieval-Model.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "1000"
        """
    )
    tb.execute()
    assert os.path.isdir("query_tower")
    assert os.path.exists("user_features.parquet")
    assert os.path.exists("item_embeddings.parquet")
    assert os.path.exists("item_features.parquet")
