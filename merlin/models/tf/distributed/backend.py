hvd = None

try:
    import horovod.tensorflow.keras as hvd  # noqa: F401
except ImportError:
    pass


if hvd:
    hvd.init()
