import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/02-Merlin-Models-and-NVTabular-integration.ipynb", execute=False)
@pytest.mark.notebook
def test_example_02_nvt_integration(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["INPUT_DATA_DIR"] = "{tmpdir}"
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
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
        p2 = patch(
            "merlin.core.utils.download_file",
            return_value=[]
        )
        p2.start()
        import numpy as np
        import pandas
        from pathlib import Path
        from merlin.datasets.synthetic import generate_data
        mock_data = generate_data(
            input="movielens-1m-raw-ratings",
            num_rows=1000
        )
        mock_data = mock_data.to_ddf().compute()
        if not isinstance(mock_data, pandas.core.frame.DataFrame):
            mock_data = mock_data.to_pandas()
        input_path = os.environ.get(
            "INPUT_DATA_DIR",
            "{tmpdir}"
        )
        path = Path(input_path) / "ml-1m"
        path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(path / "ratings.dat"),
            mock_data.values,
            delimiter='::',
            fmt='%s',
            encoding='utf-8'
        )
        """
    )
    tb.execute()
    assert tb.cell_output_text(15)[-19:] == "'TE_userId_rating']"
    metrics = tb.ref("metrics")
    assert set(metrics.keys()) == set(
        [
            "auc",
            "binary_accuracy",
            "loss",
            "loss_batch",
            "regularization_loss",
            "precision",
            "recall",
        ]
    )
