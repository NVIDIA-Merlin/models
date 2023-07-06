#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
