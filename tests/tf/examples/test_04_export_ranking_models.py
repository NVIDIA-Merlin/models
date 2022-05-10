import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.example


@testbook(REPO_ROOT / "examples/04-Exporting-ranking-models.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    tb.execute()
    assert os.path.isdir("dlrm")
    assert os.path.isdir("workflow")
