import merlin.models.tf as mm
from merlin.models.data.synthetic import SyntheticData


def test_embedding_options(testing_data: SyntheticData):
    inputs = mm.InputBlock(testing_data.schema, embedding_options=mm.EmbeddingOptions(

    ))
