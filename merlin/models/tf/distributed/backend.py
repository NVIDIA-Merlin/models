hvd = None
hvd_installed = False

dmp = None
dmp_installed = False

try:
    import horovod.tensorflow.keras as hvd  # noqa: F401

    hvd_installed = True
except ImportError:
    pass

try:
    from distributed_embeddings.python.layers import dist_model_parallel as dmp  # noqa: F401

    dmp_installed = True
except ImportError:
    pass

if hvd_installed:
    hvd.init()
