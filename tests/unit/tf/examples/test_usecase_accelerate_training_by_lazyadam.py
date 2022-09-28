# Test is currently breaks in TF 2.10

from testbook import testbook

from tests.conftest import REPO_ROOT

p = "examples/usecases/accelerate-training-of-large-embedding-tables-by-LazyAdam.ipynb"


@testbook(
    REPO_ROOT / p,
    timeout=180,
    execute=False,
)
def test_usecase_accelerate_training_by_lazyadam(tb):
    tb.inject(
        """
        import os
        os.environ["NUM_ROWS"] = "1000"
        """
    )
    tb.execute()
    model1_lazyadam = tb.ref("model1")
    model2_adam = tb.ref("model2")
    assert set(model1_lazyadam.history.history.keys()) == set(
        [
            "loss",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "precision_at_10",
            "regularization_loss",
        ]
    )
    assert set(model2_adam.history.history.keys()) == set(
        [
            "loss",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "precision_at_10",
            "regularization_loss",
        ]
    )
