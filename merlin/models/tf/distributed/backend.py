import tensorflow as tf

from merlin.core.dispatch import HAS_GPU

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

if HAS_GPU:
    try:
        from sparse_operation_kit import experiment as sok  # noqa: F401

        sok_installed = True
    except (ImportError, tf.errors.NotFoundError):
        pass


if sok_installed:
    sok.init(use_legacy_optimizer=False)
