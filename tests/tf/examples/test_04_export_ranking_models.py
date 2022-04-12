import os

import pytest
from testbook import testbook

pytestmark = pytest.mark.example


@testbook("examples/04-Exporting-ranking-models.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        """
    )
    tb.execute()
    assert os.path.isdir("dlrm")
    assert os.path.isdir("workflow")
