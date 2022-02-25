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

import nvtabular as nvt
from merlin.schema import Tags

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData


def test_topk_index(ecommerce_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(ecommerce_data.schema, query_tower=ml.MLPBlock([64, 128]))

    model: ml.RetrievalModel = two_tower.connect(
        ml.ItemRetrievalTask(ecommerce_data.schema, target_name="click", metrics=[])
    )
    model.compile(run_eagerly=True, optimizer="adam")

    dataset = ecommerce_data.tf_dataloader(batch_size=50)
    model.fit(dataset, epochs=1)

    item_features = ecommerce_data.schema.select_by_tag(Tags.ITEM).column_names
    item_dataset = ecommerce_data.dataframe[item_features].drop_duplicates("item_id")
    item_dataset = nvt.Dataset(item_dataset)

    topk_index = ml.TopKIndexBlock.from_block(two_tower.item_block(), data=item_dataset)
    recommender = two_tower.query_block().connect(topk_index)

    batch = next(iter(dataset))[0]
    _, out = recommender(batch, k=20)
    assert out.shape[-1] == 20
