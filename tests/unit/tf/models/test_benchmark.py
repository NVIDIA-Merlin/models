# #
# # Copyright (c) 2021, NVIDIA CORPORATION.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#

import merlin.models.tf as ml
from merlin.models.tf.utils import testing_utils


def test_ncf_model_single_task_from_pred_task(ecommerce_data, num_epochs=5, run_eagerly=True):
    model = ml.benchmark.NCFModel(
        ecommerce_data.schema,
        embedding_dim=64,
        mlp_block=ml.MLPBlock([64]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(ecommerce_data, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(ecommerce_data, batch_size=50, return_dict=True)
    testing_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )
