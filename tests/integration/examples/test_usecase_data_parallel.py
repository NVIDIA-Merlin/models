import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/usecases/multi-gpu-data-parallel-training.ipynb",
    execute=False,
    timeout=180,
)
@pytest.mark.notebook
def test_usecase_data_parallel(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["DATA_FOLDER"] = "{tmpdir}"
        os.environ["NUM_GPUs"]="1"
        """
    )
    tb.execute()
