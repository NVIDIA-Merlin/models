import pytorch_lightning as pl
import torch

from merlin.dataloader.torch import Loader
from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.models.base import Model
from merlin.models.torch.outputs.regression import RegressionOutput
from merlin.schema import Schema


def test_simple_regression_mlp(testing_data):
    schema: Schema = testing_data.schema.without("categories")
    model = Model(
        TabularInputBlock(schema),
        MLPBlock([10, 10]),
        RegressionOutput(),
    )

    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)
    loader = Loader(testing_data, batch_size=2, shuffle=False)

    # Initialize the model parameters
    model.initialize(loader)

    trainer.fit(model, loader)

    a = 5
