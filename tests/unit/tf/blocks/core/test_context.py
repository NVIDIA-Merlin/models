import tensorflow as tf

from merlin.io import Dataset
import merlin.models.tf as mm
from merlin.models.tf.blocks.core.context import Context


def test_context(music_streaming_data: Dataset):
    schema = music_streaming_data.schema
    features, targets = mm.sample_batch(
        music_streaming_data, batch_size=10, to_dense=True
    )

    context = Context(schema, features, targets)
    a = 5

def test_context_in_model(music_streaming_data: Dataset):
    class Dummy(tf.keras.layers.Layer):
        @tf.function
        def call(self, inputs, context):
            return context.features

        def compute_output_shape(self, input_shape):
            return input_shape

    model = mm.Model(
        Dummy(),
        mm.InputBlock(music_streaming_data.schema),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=True)
    model.fit(music_streaming_data, epochs=1, batch_size=10)