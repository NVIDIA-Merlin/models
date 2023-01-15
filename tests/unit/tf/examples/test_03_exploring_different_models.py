import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/03-Exploring-different-models.ipynb", execute=False)
@pytest.mark.notebook
def test_example_03_exploring_different_models(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["DATA_FOLDER"] = "{tmpdir}"
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
            "loss_batch",
            "regularization_loss",
        ]
    )
    metrics_mlp = tb.ref("metrics_mlp")
    assert set(metrics_mlp.keys()) == set(
        [
            "auc",
            "loss",
            "loss_batch",
            "regularization_loss",
        ]
    )
    metrics_wide_n_deep = tb.ref("metrics_wide_n_deep")
    assert set(metrics_wide_n_deep.keys()) == set(
        [
            "auc",
            "loss",
            "loss_batch",
            "regularization_loss",
        ]
    )
    metrics_dlrm = tb.ref("metrics_dlrm")
    assert set(metrics_dlrm.keys()) == set(
        [
            "auc",
            "loss",
            "loss_batch",
            "regularization_loss",
        ]
    )
    metrics_dcn = tb.ref("metrics_dcn")
    assert set(metrics_dcn.keys()) == set(
        [
            "auc",
            "loss",
            "loss_batch",
            "regularization_loss",
        ]
    )
    assert os.path.isfile(os.path.join(tmpdir, "results.txt"))
    tb.execute_cell(NUM_OF_CELLS - 2)
