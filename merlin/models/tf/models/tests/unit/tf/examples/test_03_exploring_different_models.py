import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/03-Exploring-different-models.ipynb", execute=False)
def test_example_03_exploring_different_models(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, NUM_OF_CELLS - 5)))
    metrics_ncf = tb.ref("metrics_ncf")
    assert set(metrics_ncf.keys()) == set(
        [
            "auc",
            "loss",
            "regularization_loss",
        ]
    )
    metrics_mlp = tb.ref("metrics_mlp")
    assert set(metrics_mlp.keys()) == set(
        [
            "auc_1",
            "loss",
            "regularization_loss",
        ]
    )
    metrics_dlrm = tb.ref("metrics_dlrm")
    assert set(metrics_dlrm.keys()) == set(
        [
            "auc_2",
            "loss",
            "regularization_loss",
        ]
    )
    metrics_dcn = tb.ref("metrics_dcn")
    assert set(metrics_dcn.keys()) == set(
        [
            "auc_3",
            "loss",
            "regularization_loss",
        ]
    )
    assert os.path.isfile("results.txt")
    tb.execute_cell(NUM_OF_CELLS - 2)
