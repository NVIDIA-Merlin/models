import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.predictions.dot_product import DotProductCategoricalPrediction
from merlin.models.tf.predictions.sampling.in_batch import InBatchSampler
from merlin.models.tf.utils import testing_utils


def test_dot_product_prediction(ecommerce_data: Dataset):
    model = mm.Model(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        DotProductCategoricalPrediction(ecommerce_data.schema, negative_samplers=InBatchSampler()),
    )

    _, history = testing_utils.model_test(model, ecommerce_data)
