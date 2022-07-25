from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT / "examples/usecases/entertainment-with-pretrained-embeddings.ipynb", execute=False
)
def test_usecase_pretrained_embeddings(tb):
    tb.execute()
    model = tb.ref("model")
    assert set(model.history.history.keys()) == set(["auc", "loss", "regularization_loss"])
