from tensorflow.keras.layers import Layer

from merlin.io import Dataset
from merlin.schema import Tags
import merlin.models.tf as mm
from merlin.models.tf.blocks.core.items import Items

class ToItems(Layer):
    def __init__(self, schema, **kwargs):
        super(ToItems, self).__init__(**kwargs)
        self.schema = schema

    def call(self, inputs):
        return Items.from_schema(self.schema, inputs)


def test_simple_creation_of_items(music_streaming_data: Dataset):
    schema = music_streaming_data.schema
    features = mm.sample_batch(
        music_streaming_data, batch_size=10, include_targets=False, to_dense=True
    )

    items = Items.from_schema(schema, features)

    id_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    assert features[id_name].ref() == items.ids.ref()

    metadata_names = schema.select_by_tag(Tags.ITEM).remove_by_tag(Tags.ITEM_ID)
    assert all(name in items.metadata for name in metadata_names.column_names)

    shape = items.shape

    to_items = ToItems(schema)
    out = to_items(features)

    a = 5