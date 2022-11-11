from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/02-Merlin-Models-and-NVTabular-integration.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["INPUT_DATA_DIR"] = "/raid/data/movielens"
        """
    )
    tb.execute()
    assert tb.cell_output_text(15)[-19:] == "'TE_userId_rating']"
    metrics = tb.ref("metrics")
    assert sorted(list(metrics.keys())) == [
        "loss",
        "loss_batch",
        "rating_binary/binary_classification_task/auc",
        "rating_binary/binary_classification_task/binary_accuracy",
        "rating_binary/binary_classification_task/precision",
        "rating_binary/binary_classification_task/recall",
        "regularization_loss",
    ]
