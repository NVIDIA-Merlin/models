# Test is currently breaks in TF 2.10

from testbook import testbook

from tests.conftest import REPO_ROOT

p = "examples/usecases/ecommerce-session-based-next-item-prediction-for-fashion.ipynb"


@testbook(
    REPO_ROOT / p,
    timeout=180,
    execute=False,
)
def test_usecase_ecommerce_session_based(tb):
    tb.inject(
        """
        import os
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
        mock_train, mock_valid = generate_data(
            input="dressipi2022-preprocessed",
            num_rows=10000,
            set_sizes=(0.8, 0.2)
        )
        p1 = patch(
            "merlin.datasets.ecommerce.get_dressipi2022",
            return_value=[mock_train, mock_valid]
        )
        p1.start()
        os.environ["DATA_FOLDER"] = "/tmp/dressipi2022/"
        os.environ["EPOCHS"] = "1"
        """
    )
    tb.execute()
    metrics_mlp = tb.ref("metrics_mlp")
    assert set(metrics_mlp.keys()) == set(
        [
            "loss",
            "recall_at_100",
            "mrr_at_100",
            "ndcg_at_100",
            "map_at_100",
            "precision_at_100",
            "regularization_loss",
        ]
    )
    metrics_bi_lstm = tb.ref("metrics_bi_lstm")
    assert set(metrics_bi_lstm.keys()) == set(
        [
            "loss",
            "recall_at_100",
            "mrr_at_100",
            "ndcg_at_100",
            "map_at_100",
            "precision_at_100",
            "regularization_loss",
        ]
    )
