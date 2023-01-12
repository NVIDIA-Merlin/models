import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

optuna = pytest.importorskip("optuna")


@testbook(
    REPO_ROOT / "examples/usecases/retrieval-with-hyperparameter-optimization.ipynb", execute=False
)
@pytest.mark.notebook
def test_usecase_pretrained_embeddings(tb):
    tb.inject(
        """
        from merlin.datasets.synthetic import generate_data
        ds = generate_data('transactions', 10000)
        df = ds.compute()
        from datetime import datetime, timedelta
        import random
        def generate_date():
            date = datetime.today()
            if random.randint(0, 1):
                date -= timedelta(days=7)
            return date
        t_dat = [generate_date() for _ in range(df.shape[0])]
        df['t_dat'] = t_dat
        df.to_csv('transactions_train.csv')
        """
    )
    tb.cells[
        27
    ].source = (
        "search_space = {'learning_rate': [1e-3], 'num_epochs': [1, 2], 'embedding_dim': [16]}"
    )
    tb.execute()
    study = tb.ref("study")
    assert set(study.best_params.keys()) == set(["learning_rate", "num_epochs", "embedding_dim"])
