from testbook import testbook


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
        "rating_binary/binary_classification_task/precision",
        "rating_binary/binary_classification_task/recall",
        "rating_binary/binary_classification_task/binary_accuracy",
        "rating_binary/binary_classification_task/auc",
        "loss",
        "regularization_loss",
        "total_loss",
    ]
