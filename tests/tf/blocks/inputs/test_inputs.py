import merlin.models.tf as mm
from merlin.models.data.synthetic import SyntheticData


def test_embedding_options(testing_data: SyntheticData):
    inputs = mm.InputBlock(
        testing_data.schema,
        embedding_options=mm.EmbeddingOptions(
            testing_data.schema,
            default_embedding_dim=10,
        ),
    )

    assert inputs is not None
