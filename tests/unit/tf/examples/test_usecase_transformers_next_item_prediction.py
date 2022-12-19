import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("transformers")


@testbook(
    REPO_ROOT / "examples/usecases/transformers-next-item-prediction.ipynb",
    timeout=180,
    execute=False,
)
def test_next_item_prediction(tb):
    tb.inject(
        """
        import os, random
        from datetime import datetime, timedelta
        from merlin.datasets.synthetic import generate_data
        ds = generate_data('booking.com-raw', 10000)
        df = ds.compute()
        def generate_date():
            date = datetime.today()
            if random.randint(0, 1):
                date -= timedelta(days=7)
            return date
        df['checkin'] = [generate_date() for _ in range(df.shape[0])]
        df['checkout'] = [generate_date() for _ in range(df.shape[0])]
        df.to_csv('/tmp/train_set.csv')
        """
    )
    tb.cells[4].source = tb.cells[4].source.replace("get_booking('/workspace/data')", "")
    tb.cells[4].source = tb.cells[4].source.replace(
        "read_csv('/workspace/data/train_set.csv'", "read_csv('/tmp/train_set.csv'"
    )
    tb.cells[28].source = tb.cells[28].source.replace("d_model=64", "d_model=40")
    tb.cells[30].source = tb.cells[30].source.replace("epochs=5", "epochs=1")
    tb.execute()
