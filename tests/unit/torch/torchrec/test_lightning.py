# import os

# import pytorch_lightning as pl
# from lightning.fabric import Fabric

# from merlin.dataloader.torch import Loader
# from merlin.io import Dataset
# from merlin.models.torch.blocks.mlp import MLPBlock
# from merlin.models.torch.combinators import ParallelBlock
# from merlin.models.torch.inputs.base import TabularInputBlock
# from merlin.models.torch.torchrec.inputs import EmbeddingBagCollection
# from merlin.models.torch.outputs.classification import BinaryOutput
# from merlin.models.torch.outputs.regression import RegressionOutput
# from merlin.models.torch.torchrec.lightning import TorchrecModel, TorchrecStrategy
# from merlin.schema import Schema, Tags


# class TestTorcrecModel:
#     def test_simple_regression_mlp(self, music_streaming_data):
#         # Multi-hot is not supported yet, TODO: add support for multi-hot
#         # schema: Schema = music_streaming_data.schema.without(["user_genres", "like", "item_genres"])
#         # music_streaming_data.schema = schema
#         schema = music_streaming_data.schema

#         fabric = Fabric(
#             # strategy=TorchrecStrategy()
#         )
#         fabric.launch()

#         embeddings = EmbeddingBagCollection(schema.select_by_tag(Tags.CATEGORICAL))
#         model = TorchrecModel(
#             TabularInputBlock(schema, categorical=embeddings),
#             MLPBlock([10, 10]),
#             ParallelBlock(
#                 RegressionOutput(schema.select_by_name("play_percentage").first),
#                 BinaryOutput("click"),
#             ),
#         )

#         assert model.input_schema == schema.excluding_by_tag(TabularInputBlock.TAGS_TO_EXCLUDE)
#         assert set(model.output_schema.column_names) == set(
#             schema.select_by_tag(Tags.TARGET).column_names
#         )

#         trainer = pl.Trainer(
#             max_epochs=1,
#             strategy=TorchrecStrategy(),
#             devices=os.environ.get("LOCAL_WORLD_SIZE", 1),
#         )
#         loader = Loader(music_streaming_data, batch_size=2, shuffle=False)

#         # Initialize the model parameters
#         model.initialize(loader)

#         trainer.fit(model, loader)

#         assert trainer.state.status.value == "finished"

#         predictions = model.batch_predict(music_streaming_data, batch_size=10)
#         assert isinstance(predictions, Dataset)

#         ddf = predictions.compute(scheduler="synchronous")
#         assert len(ddf) == 100
#         assert len(ddf.columns) == 15
