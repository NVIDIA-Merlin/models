import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/06-Define-your-own-architecture-with-Merlin-Models.ipynb",
    timeout=180,
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
    assert sorted(list(metrics.keys())) == [
        "loss",
        "rating_binary/binary_classification_task/auc",
        "regularization_loss",
        "total_loss",
    ]
    assert os.path.isdir("custom_dlrm")
