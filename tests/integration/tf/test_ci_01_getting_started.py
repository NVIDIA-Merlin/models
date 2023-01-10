from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/01-Getting-started.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["INPUT_DATA_DIR"] = "/raid/data/movielens"
        """
    )
    tb.execute()
    metrics = tb.ref("metrics")
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "precision",
            "recall",
            "binary_accuracy",
            "auc",
            "regularization_loss",
            "loss_batch",
        ]
    )
