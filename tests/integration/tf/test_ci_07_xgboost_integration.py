from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/07-Train-a-third-party-model-using-the-Merlin-Models-API.ipynb",
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
    lightfm_metrics = tb.ref("lightfm_metrics")
    implicit_metrics = tb.ref("implicit_metrics")

    assert metrics.keys() == {"logloss"}
    assert metrics["logloss"] < 0.65
    assert sorted(lightfm_metrics) == ["auc", "precisions@10"]
    assert sorted(implicit_metrics) == ["auc@10", "map@10", "ndcg@10", "precision@10"]
