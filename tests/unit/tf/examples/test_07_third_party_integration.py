from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/07-Train-a-third-party-model-using-the-Merlin-Models-API.ipynb",
    execute=False,
)
def test_func(tb):
    tb.inject(
        """
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
        mock_train, mock_valid = generate_data(
            input="movielens-100k",
            num_rows=1000,
            set_sizes=(0.8, 0.2)
        )
        p1 = patch(
            "merlin.datasets.entertainment.get_movielens",
            return_value=[mock_train, mock_valid]
        )
        p1.start()
        """
    )
    tb.cells.pop(26)
    tb.execute()
    xgboost_metrics = tb.ref("metrics")
    implicit_metrics = tb.ref("implicit_metrics")

    assert xgboost_metrics.keys() == {"logloss"}
    assert sorted(implicit_metrics) == ["auc@10", "map@10", "ndcg@10", "precision@10"]
