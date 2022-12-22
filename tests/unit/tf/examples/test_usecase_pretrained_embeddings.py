import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/usecases/entertainment-with-pretrained-embeddings.ipynb", execute=False
)
@pytest.mark.notebook
def test_usecase_pretrained_embeddings(tb):
    tb.execute()
    history = tb.ref("history")
    assert set(history.keys()) == set(["auc", "loss", "loss_batch", "regularization_loss"])
