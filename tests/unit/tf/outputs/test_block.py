import pytest

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_output(ecommerce_data: Dataset, run_eagerly: bool):
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([4]),
        mm.OutputBlock(ecommerce_data.schema),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)
