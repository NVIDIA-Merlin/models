import pytorch_lightning as pl

from merlin.dataloader.torch import Loader
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.classification import BinaryOutput
from merlin.models.torch.outputs.regression import RegressionOutput
from merlin.schema import Schema, Tags


def test_simple_regression_mlp(music_streaming_data):
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
