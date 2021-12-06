import pytest

from merlin_models.data.synthetic import SyntheticData

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


class DummyFeaturesBlock(ml.FeaturesBlock):
    def call_features(self, features, **kwargs):
        self.position = features["position"]

    def call(self, inputs, **kwargs):
        position = self.position

        return inputs * tf.cast(tf.reduce_mean(position), tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


def test_block_context(ecommerce_data: SyntheticData):
    inputs = ml.inputs(ecommerce_data.schema)
    model = inputs.connect(ml.MLPBlock([64]), DummyFeaturesBlock())

    out = model(ecommerce_data.tf_tensor_dict)

    assert out.shape[-1] == 64
