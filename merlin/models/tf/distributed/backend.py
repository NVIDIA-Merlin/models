import tensorflow as tf

from merlin.core.dispatch import HAS_GPU

hvd = None
multi_gpu = False

if HAS_GPU:
    try:
        import horovod.tensorflow.keras as hvd

        multi_gpu = True
    except ImportError:
        pass


gpus = None

if multi_gpu:
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
