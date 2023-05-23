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
    history_with_embeddings = tb.ref("history_with_embeddings")
    assert set(history_with_embeddings.keys()) == set(
        ["auc_1", "loss", "loss_batch", "regularization_loss"]
    )
