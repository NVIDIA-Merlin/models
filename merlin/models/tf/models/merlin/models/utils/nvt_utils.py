import logging


def require_nvt():
    try:
        import nvtabular as nvt  # noqa

        backend = None
        try:
            import tensorflow as tf

            backend = tf

            if tf.config.list_physical_devices("GPU"):
                _check_nvt_gpu()

        except ImportError:
            pass

        if not backend:
            try:
                import torch

                backend = torch

                if torch.cuda.is_available():
                    _check_nvt_gpu()
            except ImportError:
                pass

    except ImportError:
        raise ImportError(
            "nvtabular is required for this feature.",
            "Please install it with `pip install nvtabular`.",
        )


def _check_nvt_gpu():
    try:
        import cudf  # noqa
    except ImportError:
        logging.warning(
            "A GPU was detected but rapids is not installed.",
            "NVTabular will not be able to use GPU.",
            "Look at the documentation for more information at rapids.ai",
        )
