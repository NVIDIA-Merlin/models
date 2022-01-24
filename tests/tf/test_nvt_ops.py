import nvtabular as nvt
import pytest

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData
from merlin_models.tf.nvt_ops import ItemEmbeddings, TFModelEncode, UserEmbeddings


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_encode(ecommerce_data: SyntheticData, run_eagerly):
    prediction_task = ml.PredictionTasks(ecommerce_data.schema)

    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(prediction_task)
    model.compile(run_eagerly=run_eagerly, optimizer="adam")

    dataset = ecommerce_data.tf_dataloader(batch_size=50)
    model.fit(dataset, epochs=1)

    data = TFModelEncode(model).fit_transform(nvt.Dataset(ecommerce_data.dataframe))
    ddf = data.to_ddf().compute(scheduler="synchronous")

    assert len(list(ddf.columns)) == 25
    assert all([task in list(ddf.columns) for task in model.block.last.task_names])


def test_two_tower_embedding_extraction(ecommerce_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(ecommerce_data.schema, query_tower=ml.MLPBlock([64, 128]))

    model = two_tower.connect(ml.ItemRetrievalTask(target_name="click", metrics=[]))
    model.compile(run_eagerly=True, optimizer="adam")

    dataset = ecommerce_data.tf_dataloader(batch_size=50)
    model.fit(dataset, epochs=1)

    item_embs = ItemEmbeddings(model, dim=128).fit_transform(nvt.Dataset(ecommerce_data.dataframe))
    item_embs_ddf = item_embs.to_ddf().compute(scheduler="synchronous")

    assert len(list(item_embs_ddf.columns)) == 23 + 128

    user_embs = UserEmbeddings(model, dim=128).fit_transform(nvt.Dataset(ecommerce_data.dataframe))
    user_embs_ddf = user_embs.to_ddf().compute(scheduler="synchronous")

    assert len(list(user_embs_ddf.columns)) == 23 + 128
