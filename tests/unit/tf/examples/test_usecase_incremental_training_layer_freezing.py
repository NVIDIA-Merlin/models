# Test is currently breaks in TF 2.10

from testbook import testbook

from tests.conftest import REPO_ROOT

p = "examples/usecases/incremental-training-with-layer-freezing.ipynb"


@testbook(
    REPO_ROOT / p,
    timeout=180,
    execute=False,
)
def test_usecase_incremental_training_layer_freezing(tb):
    tb.inject(
        """
        import os
        os.environ["NUM_ROWS"] = "1000"
        """
    )
    tb.execute()
    model = tb.ref("model")
    assert set(model.history.history.keys()) == set(
        [
            "loss",
            "loss_batch",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "precision_at_10",
            "regularization_loss",
        ]
    )
