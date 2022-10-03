def test_import():
    import horovod.tensorflow.keras as hvd

    assert hvd is not None
