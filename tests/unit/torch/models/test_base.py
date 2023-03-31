import pytorch_lightning as pl
import torch

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.inputs.encoder import Encoder
from merlin.models.torch.models.base import Model, RetrievalModel
from merlin.models.torch.outputs.classification import BinaryOutput
from merlin.models.torch.outputs.contrastive import ContrastiveOutput
from merlin.models.torch.outputs.regression import RegressionOutput
from merlin.schema import Schema, Tags


class TestModel:
    def test_simple_regression_mlp(self, music_streaming_data):
        # Multi-hot is not supported yet, TODO: add support for multi-hot
        schema: Schema = music_streaming_data.schema.without(["user_genres", "like", "item_genres"])
        music_streaming_data.schema = schema

        model = Model(
            TabularInputBlock(schema),
            MLPBlock([10, 10]),
            ParallelBlock(
                RegressionOutput(schema.select_by_name("play_percentage").first),
                BinaryOutput("click"),
            ),
        )

        assert model.input_schema == schema.excluding_by_tag(TabularInputBlock.TAGS_TO_EXCLUDE)
        assert set(model.output_schema.column_names) == set(
            schema.select_by_tag(Tags.TARGET).column_names
        )

        trainer = pl.Trainer(max_epochs=1, devices=[0])
        loader = Loader(music_streaming_data, batch_size=2, shuffle=False)

        # Initialize the model parameters
        model.initialize(loader)

        # import torch
        # if hasattr(torch, "compile"):
        #     model = torch.compile(model)

        trainer.fit(model, loader)

        assert trainer.state.status.value == "finished"

        predictions = model.batch_predict(music_streaming_data, batch_size=10)
        assert isinstance(predictions, Dataset)

        ddf = predictions.compute(scheduler="synchronous")
        assert len(ddf) == 100
        assert len(ddf.columns) == 15


class TestRetrievalModel:
    def test_mf(self, user_id_col_schema, item_id_col_schema):
        model = RetrievalModel(
            query=EmbeddingTable(10, user_id_col_schema),
            output=ContrastiveOutput(item_id_col_schema),
        )

        user_id = torch.tensor([0, 1, 2])
        item_id = torch.tensor([1, 2, 3])

        outputs, targets = model(user_id, targets=item_id)
        assert outputs.shape == (3, 4)
        assert targets.shape == (3, 4)
        assert torch.equal(targets[:, 0], torch.ones(3))
        assert torch.equal(targets[:, 1:], torch.zeros([3, 3]))

        user_embs = model.query_embeddings(gpu=False)
        user_embs_ddf = user_embs.compute(scheduler="synchronous")
        assert isinstance(user_embs, Dataset)
        assert len(user_embs_ddf) == 21
        assert len(user_embs_ddf.columns) == 10

        candidate_embs = model.candidate_embeddings(gpu=False)
        candidate_embs_ddf = candidate_embs.compute(scheduler="synchronous")
        assert isinstance(candidate_embs, Dataset)
        assert len(candidate_embs_ddf) == 11
        assert len(candidate_embs_ddf.columns) == 10

    def test_two_tower(self, user_id_col_schema, item_id_col_schema):
        model = RetrievalModel(
            query=Encoder(Schema([user_id_col_schema])),
            candidate=Encoder(Schema([item_id_col_schema])),
            output=ContrastiveOutput(item_id_col_schema),
        )

        assert isinstance(model.query, Encoder)
