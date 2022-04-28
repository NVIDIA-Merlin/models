import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/03-Exploring-different-models.ipynb", execute=False)
def test_func(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "999"
        """
    )
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, NUM_OF_CELLS - 3)))
    metrics_ncf = tb.ref("metrics_ncf")
    assert sorted(list(metrics_ncf.keys())) == [
        "auc",
        "loss",
        "regularization_loss",
        "total_loss",
    ]
    metrics_mlp = tb.ref("metrics_mlp")
    assert sorted(list(metrics_mlp.keys())) == [
        "auc_1",
        "loss",
        "regularization_loss",
        "total_loss",
    ]
    metrics_dlrm = tb.ref("metrics_dlrm")
    assert sorted(list(metrics_dlrm.keys())) == [
        "auc_2",
        "loss",
        "regularization_loss",
        "total_loss",
    ]
    metrics_dcn = tb.ref("metrics_dcn")
    assert sorted(list(metrics_dcn.keys())) == [
        "auc_3",
        "loss",
        "regularization_loss",
        "total_loss",
    ]
    assert os.path.isfile("results.txt")
    tb.execute_cell(NUM_OF_CELLS)
