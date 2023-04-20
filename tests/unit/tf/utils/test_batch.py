import pandas as pd
import pytest

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.models.base import get_task_names_from_outputs


@pytest.mark.parametrize(
    "output_type", ["prediction_tasks_v1", "model_outputs_v2-single", "model_outputs_v2-multiple"]
)
@pytest.mark.parametrize("run_eagerly", [True, False])
def test_model_encode(ecommerce_data: Dataset, output_type: str, run_eagerly):
    if output_type == "prediction_tasks_v1":
        output_block = ml.PredictionTasks(ecommerce_data.schema)
    elif output_type == "model_outputs_v2-single":
        output_block = ml.BinaryOutput("click")
    elif output_type == "model_outputs_v2-multiple":
        output_block = ml.OutputBlock(ecommerce_data.schema)
    output_task_names = get_task_names_from_outputs(output_block)

    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([2]),
        output_block,
    )

    model.compile(run_eagerly=run_eagerly, optimizer="adam")
    model.fit(ecommerce_data, batch_size=50, epochs=1, steps_per_epoch=1)
    data = model.batch_predict(ecommerce_data, batch_size=10)
    ddf = data.compute(scheduler="synchronous")

    assert len(list(ddf.columns)) == 26 if output_type == "model_outputs_v2-single" else 27
    assert all([task in list(ddf.columns) for task in output_task_names])


def test_two_tower_embedding_extraction(ecommerce_data: Dataset):
    model = ml.RetrievalModel(
        ml.TwoTowerBlock(ecommerce_data.schema, query_tower=ml.MLPBlock([2])),
        ml.ItemRetrievalTask(ecommerce_data.schema, target_name="click"),
    )
    model.compile(run_eagerly=True, optimizer="adam")
    model.fit(ecommerce_data, batch_size=50, epochs=1)

    item_embs = model.item_embeddings(ecommerce_data, 10)
    item_embs_ddf = item_embs.compute(scheduler="synchronous")

    assert len(list(item_embs_ddf.columns)) == 5 + 2

    user_embs = model.query_embeddings(ecommerce_data, 10)
    user_embs_ddf = user_embs.compute(scheduler="synchronous")

    assert len(list(user_embs_ddf.columns)) == 13 + 2


def test_two_tower_extracted_embeddings_are_equal(ecommerce_data: Dataset):
    import numpy as np

    model = ml.RetrievalModel(
        ml.TwoTowerBlock(ecommerce_data.schema, query_tower=ml.MLPBlock([2])),
        ml.ItemRetrievalTask(ecommerce_data.schema, target_name="click"),
    )
    model.compile(run_eagerly=True, optimizer="adam")
    model.fit(ecommerce_data, batch_size=50, epochs=1)

    item_embs_1 = model.item_embeddings(ecommerce_data, batch_size=10).compute()
    item_embs_2 = model.item_embeddings(ecommerce_data, batch_size=10).compute()

    if not isinstance(item_embs_1, pd.DataFrame):
        item_embs_1 = item_embs_1.to_pandas()
        item_embs_2 = item_embs_2.to_pandas()

    np.testing.assert_array_equal(item_embs_1.values, item_embs_2.values)
