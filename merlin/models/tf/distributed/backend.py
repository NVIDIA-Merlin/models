hvd = None
hvd_installed = False

sok = None
sok_installed = False


try:
    import horovod.tensorflow.keras as hvd  # noqa: F401

    hvd_installed = True
except ImportError:
    pass


if hvd_installed:
    hvd.init()

try:
    from sparse_operation_kit import experiment as sok  # noqa: F401

    sok_installed = True
except ImportError:
    pass


if sok_installed:
    sok.init()
