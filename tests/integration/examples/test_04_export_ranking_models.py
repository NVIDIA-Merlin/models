import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/04-Exporting-ranking-models.ipynb", execute=False)
@pytest.mark.notebook
def test_example_04_exporting_ranking_models(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["DATA_FOLDER"] = "{tmpdir}"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    tb.execute()
    assert os.path.isdir(os.path.join(tmpdir, "dlrm"))
    assert os.path.isdir(os.path.join(tmpdir, "workflow"))
