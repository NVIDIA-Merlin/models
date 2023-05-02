# TODO: Test is currently breaks in TF 2.10

import pytest
from testbook import testbook

from merlin.core.dispatch import HAS_GPU
from tests.conftest import REPO_ROOT

p = "examples/usecases/accelerate-training-of-large-embedding-tables-by-LazyAdam.ipynb"


@testbook(
    REPO_ROOT / p,
    timeout=180,
    execute=False,
)
@pytest.mark.notebook
@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
def test_usecase_accelerate_training_by_lazyadam(tb):
    tb.inject(
        """
        import os
        os.environ["NUM_ROWS"] = "1000"
        os.environ["dataset_name"] = "e-commerce"
        """
    )
    tb.execute()
    model1_lazyadam = tb.ref("model1")
    model2_adam = tb.ref("model2")
    assert set(model1_lazyadam.history.history.keys()) == set(
        [
            "loss",
            "loss_batch",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "precision_at_10",
            "regularization_loss",
        ]
    )
    assert set(model2_adam.history.history.keys()) == set(
        [
            "loss",
            "loss_batch",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "map_at_10",
            "precision_at_10",
            "regularization_loss",
        ]
    )
