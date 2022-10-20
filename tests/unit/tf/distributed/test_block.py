import importlib

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("horovod") is None, reason="This unit test requires horovod"
)
def test_import():
    from merlin.models.tf.distributed.backend import hvd

    assert hvd is not None
