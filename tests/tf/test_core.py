import pytest

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


class DummyFeaturesBlock(ml.Block):
    def call_features(self, features, **kwargs):
        self.items = features[str(Tag.ITEM_ID)]

    def call(self, inputs, **kwargs):
        item_embedding_table = self.context.get_embedding(Tag.ITEM_ID)
        item_embeddings = tf.gather(item_embedding_table, tf.cast(self.items, tf.int32))

        return inputs * item_embeddings

    def compute_output_shape(self, input_shape):
        return input_shape


def test_block_context(ecommerce_data: SyntheticData):
    inputs = ml.inputs(ecommerce_data.schema)
    model = inputs.connect(ml.MLPBlock([64]), DummyFeaturesBlock())

    out = model(ecommerce_data.tf_tensor_dict)

    assert out.shape[-1] == 64
