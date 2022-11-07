from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/usecases/transformers-next-item-prediction.ipynb", execute=False)
def test_usecase_pretrained_embeddings(tb):
    tb.inject(
        """
        import os, random
        from datetime import datetime, timedelta
        from merlin.datasets.synthetic import generate_data
        ds = generate_data('bookingdotcom', 10000)
        if not os.path.exists('ml-dataset-mdt'): os.mkdir('ml-dataset-mdt')
        df = ds.compute()
        def generate_date():
            date = datetime.today()
            if random.randint(0, 1):
                date -= timedelta(days=7)
            return date
        df['checkin'] = [generate_date() for _ in range(df.shape[0])]
        df.to_csv('ml-dataset-mdt/train_set.csv')
        """
    )
    tb.cells[31].source = tb.cells[31].source.replace("d_model=40", "d_model=16")
    tb.cells[33].source = tb.cells[33].source.replace("epochs=5", "epochs=1")
    tb.execute()
