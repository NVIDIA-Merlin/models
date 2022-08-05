from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/07-Train-an-xgboost-model-using-the-Merlin-Models-API.ipynb",
    execute=False,
)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["INPUT_DATA_DIR"] = "/raid/data/movielens"
        """
    )
    tb.execute()
    metrics = tb.ref("metrics")
    assert metrics.keys() == {"logloss"}
    assert metrics["logloss"] < 0.65
