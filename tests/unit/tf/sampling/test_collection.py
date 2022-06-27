from tensorflow.keras.layers import Layer

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.sampling.collection import ItemCollection
from merlin.schema import Tags


class ToItemCollection(Layer):
    def __init__(self, schema, **kwargs):
        super(ToItemCollection, self).__init__(**kwargs)
        self.schema = schema

    def call(self, inputs) -> ItemCollection:
        return ItemCollection.from_features(self.schema, inputs)


def test_simple_creation_of_items(music_streaming_data: Dataset):
    schema = music_streaming_data.schema
    features = mm.sample_batch(
        music_streaming_data, batch_size=10, include_targets=False, to_dense=True
    )

    items = ItemCollection.from_features(schema, features)

    id_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    assert features[id_name].ref() == items.ids.ref()

    metadata_names = schema.select_by_tag(Tags.ITEM).remove_by_tag(Tags.ITEM_ID)
    assert all(name in items.features for name in metadata_names.column_names)

    assert all(name in items.shape for name in metadata_names.column_names)

    to_items = ToItemCollection(schema)
    out = to_items(features)

    assert out is not None
