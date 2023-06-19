import shutil

import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("transformers")
utils = pytest.importorskip("merlin.systems.triton.utils")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@testbook(
    REPO_ROOT
    / "examples/usecases/transformers-next-item-prediction-with-pretrained-embeddings.ipynb",
    timeout=720,
    execute=False,
)
@pytest.mark.notebook
def test_next_item_prediction(tb, tmpdir):
    tb.inject(
        f"""
        import os, random
        os.environ["OUTPUT_DATA_DIR"] = "{tmpdir}"
        os.environ["NUM_EPOCHS"] = "1"
        os.environ["NUM_EXAMPLES"] = "10_000"
        """
    )
    tb.execute_cell(list(range(0, 42)))

    with utils.run_triton_server(f"{tmpdir}/ensemble", grpc_port=8001):
        tb.execute_cell(list(range(42, len(tb.cells))))

    predicted_hashed_url_id = tb.ref("predicted_hashed_url_id").item()
    assert predicted_hashed_url_id >= 0 and predicted_hashed_url_id <= 1002
