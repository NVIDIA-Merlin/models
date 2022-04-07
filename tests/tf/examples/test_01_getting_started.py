import pytest
from testbook import testbook

pytestmark = pytest.mark.example


@testbook("examples/01-Getting-started.ipynb", execute=False)
def test_func(tb):
    tb.execute_cell("imports")

    # For this test we use synthetic data instead of real movielens data.
    tb.inject(
        """
        from merlin.datasets.synthetic import generate_data
        train, valid = generate_data(
            input="movielens-1m",
            num_rows=1000,
            set_sizes=(0.8, 0.2)
        )
        """
    )

    tb.execute_cell("model")
    tb.execute_cell("fit")
    tb.execute_cell("evaluate")

    metrics = tb.ref("metrics")

    assert sorted(list(metrics.keys())) == [
        "loss",
        "rating_binary/binary_classification_task/auc",
        "rating_binary/binary_classification_task/binary_accuracy",
        "rating_binary/binary_classification_task/precision",
        "rating_binary/binary_classification_task/recall",
        "regularization_loss",
        "total_loss",
    ]
