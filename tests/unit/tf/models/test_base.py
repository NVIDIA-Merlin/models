#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest

import merlin.models.tf as ml
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(ecommerce_data: Dataset, run_eagerly):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        ml.BinaryClassificationTask("click"),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_from_block(ecommerce_data: Dataset, run_eagerly):
    embedding_options = ml.EmbeddingOptions(embedding_dim_default=32)
    model = ml.Model.from_block(
        ml.MLPBlock([64]),
        ecommerce_data.schema,
        prediction_tasks=ml.BinaryClassificationTask("click"),
        embedding_options=embedding_options,
    )

    assert all(
        [f.table.dim == 32 for f in list(model.block.inputs["categorical"].feature_config.values())]
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


def test_block_from_model_with_input(ecommerce_data: Dataset):
    inputs = ml.InputBlock(ecommerce_data.schema)
    block = inputs.connect(ml.MLPBlock([64]))

    with pytest.raises(ValueError) as excinfo:
        ml.Model.from_block(
            block,
            ecommerce_data.schema,
            input_block=inputs,
        )
    assert "The block already includes an InputBlock" in str(excinfo.value)
