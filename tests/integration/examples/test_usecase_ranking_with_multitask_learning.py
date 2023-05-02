import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/usecases/ranking_with_multitask_learning.ipynb", execute=False)
@pytest.mark.notebook
def test_usecase_ranking_with_multitask_learning(tb):
    tb.inject(
        """
        import os
        os.environ["NUM_ROWS"] = "999"
        """
    )
    tb.execute()
    metrics = tb.ref("metrics_results")
    assert set(metrics.keys()) == set(
        [
            "loss",
            "click/binary_output_loss",
            "follow/binary_output_loss",
            "like/binary_output_loss",
            "share/binary_output_loss",
            "watching_times/regression_output_loss",
            "click/binary_output/precision",
            "click/binary_output/recall",
            "click/binary_output/binary_accuracy",
            "click/binary_output/auc",
            "follow/binary_output/precision",
            "follow/binary_output/recall",
            "follow/binary_output/binary_accuracy",
            "follow/binary_output/auc",
            "like/binary_output/precision",
            "like/binary_output/recall",
            "like/binary_output/binary_accuracy",
            "like/binary_output/auc",
            "share/binary_output/precision",
            "share/binary_output/recall",
            "share/binary_output/binary_accuracy",
            "share/binary_output/auc",
            "watching_times/regression_output/root_mean_squared_error",
            "regularization_loss",
            "loss_batch",
        ]
    )
