import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/04-Exporting-ranking-models.ipynb", execute=False)
def test_example_04_exporting_ranking_models(tb):
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
