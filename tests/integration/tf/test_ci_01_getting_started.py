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
    assert sorted(list(metrics.keys())) == [
        "loss",
        "rating_binary/binary_classification_task/auc",
        "rating_binary/binary_classification_task/binary_accuracy",
        "rating_binary/binary_classification_task/precision",
        "rating_binary/binary_classification_task/recall",
        "regularization_loss",
    ]
