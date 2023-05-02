import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/05-Retrieval-Model.ipynb", execute=False)
@pytest.mark.notebook
def test_example_05_retrieval_models(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["DATA_FOLDER"] = "{tmpdir}"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    tb.execute()
    assert os.path.isdir(os.path.join(tmpdir, "query_tower"))
    assert os.path.exists(os.path.join(tmpdir, "user_features.parquet"))
    assert os.path.exists(os.path.join(tmpdir, "item_embeddings.parquet"))
    assert os.path.exists(os.path.join(tmpdir, "item_features.parquet"))
