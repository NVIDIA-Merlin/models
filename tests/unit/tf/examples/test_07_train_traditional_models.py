import pytest
from testbook import testbook

from merlin.core.dispatch import HAS_GPU
from tests.conftest import REPO_ROOT

pytest.importorskip("xgboost")
pytest.importorskip("lightfm")
pytest.importorskip("implicit")


@testbook(
    REPO_ROOT / "examples/07-Train-traditional-ML-models-using-the-Merlin-Models-API.ipynb",
    execute=False,
)
@pytest.mark.notebook
@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
def test_func(tb):
    tb.execute()
    xgboost_metrics = tb.ref("metrics")
    implicit_metrics = tb.ref("implicit_metrics")

    assert xgboost_metrics.keys() == {"logloss"}
    assert sorted(implicit_metrics) == ["auc@10", "map@10", "ndcg@10", "precision@10"]
