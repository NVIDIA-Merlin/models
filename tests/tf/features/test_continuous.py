import pytest

from merlin_models import tf as mtf

tf = pytest.importorskip("tensorflow")
# mtf = pytest.importorskip("merlin_models.tf")


def test_continuous_features(continuous_features):
    features = ["scalar_continuous"]
    con = mtf.ContinuousFeatures(features)(continuous_features)

    assert list(con.keys()) == features
