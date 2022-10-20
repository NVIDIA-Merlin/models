import os
import time

import nest_asyncio
from testbook import testbook

from tests.conftest import REPO_ROOT

p = "examples/usecases/multi-gpu-training.ipynb"
nest_asyncio.apply()


@testbook(
    REPO_ROOT / p,
    timeout=180,
    execute=False,
)
def test_usecase_multi_gpu_training(tb, tmpdir):
    tb.inject(
        f"""
        import os
        os.environ["DATA_FOLDER"] = "{tmpdir}"
        """
    )
    tb.execute()
    assert "saved_model.pb" in os.listdir(os.path.join(tmpdir, "processed"))
