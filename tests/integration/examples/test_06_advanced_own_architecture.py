import os

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/06-Define-your-own-architecture-with-Merlin-Models.ipynb", execute=False
)
@pytest.mark.notebook
def test_example_06_defining_own_architecture(tb, tmpdir):
    tb.inject(
        f"""
        import os
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
        os.environ["DATA_FOLDER"] = "{tmpdir}"
        mock_train, mock_valid = generate_data(
            input="movielens-1m",
            num_rows=1000,
            set_sizes=(0.8, 0.2)
        )
        p1 = patch(
            "merlin.datasets.entertainment.get_movielens",
            return_value=[mock_train, mock_valid]
        )
        p1.start()
        """
    )
    tb.execute()
    metrics = tb.ref("metrics")
    assert set(metrics.keys()) == set(["auc", "loss", "loss_batch", "regularization_loss"])
    assert os.path.isdir(os.path.join(tmpdir, "custom_dlrm"))
