hvd = None
hvd_installed = False

try:
    import horovod.tensorflow.keras as hvd  # noqa: F401

    hvd_installed = True
except ImportError:
    pass


if hvd_installed:
    hvd.init()
