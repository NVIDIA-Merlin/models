from merlin.core.dispatch import HAS_GPU

hvd = None

if HAS_GPU:
    try:
        import horovod.tensorflow as hvd  # noqa: F401
    except ImportError:
        pass

if hvd:
    hvd.init()
