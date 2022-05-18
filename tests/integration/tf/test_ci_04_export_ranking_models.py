import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/04-Exporting-ranking-models.ipynb", timeout=180, execute=False)
def test_func(tb):
    if not os.path.isdir("/raid/data/aliccp/raw/"):
        pytest.skip("Data not found.")
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/raid/data/aliccp/raw/"
        os.environ["NUM_ROWS"] = "999"
        os.environ["SYNTHETIC_DATA"] = "False"
        os.environ["BATCH_SIZE"] = "16384"
        """
    )
    tb.execute()
    assert os.path.isdir("dlrm")
    assert os.path.isdir("workflow")
