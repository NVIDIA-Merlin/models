#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
# from functools import lru_cache

from __future__ import absolute_import

import platform
import warnings
from pathlib import Path
from unittest.mock import patch

import distributed
import psutil
import pytest
from asvdb import BenchmarkInfo, utils

from merlin.core.utils import Distributed
from merlin.dataloader.loader_base import LoaderBase
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.utils import ci_utils

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def ecommerce_data() -> Dataset:
    return generate_data("e-commerce", num_rows=100)


@pytest.fixture
def music_streaming_data() -> Dataset:
    return generate_data("music-streaming", num_rows=100)


@pytest.fixture
def sequence_testing_data() -> Dataset:
    return generate_data("sequence-testing", num_rows=100)


@pytest.fixture
def social_data() -> Dataset:
    return generate_data("social", num_rows=100)


@pytest.fixture
def criteo_data() -> Dataset:
    return generate_data("criteo", num_rows=100)


@pytest.fixture
def testing_data() -> Dataset:
    data = generate_data("testing", num_rows=100)
    data.schema = data.schema.without(["session_id", "session_start", "day_idx"])

    return data


@pytest.fixture(scope="module")
def dask_client() -> distributed.Client:
    with Distributed() as dist:
        yield dist.client


try:
    import tensorflow as tf  # noqa

    from tests.unit.tf._conftest import *  # noqa
except ImportError:
    pass

try:
    import torchmetrics  # noqa

    from tests.unit.torch._conftest import *  # noqa
except ModuleNotFoundError:
    pass


def pytest_collection_modifyitems(items):
    try:
        changed_backends = ci_utils.get_changed_backends()
    except Exception as e:
        warnings.warn(
            f"Running all tests because CI utils failed to detect backend changes: {e}", UserWarning
        )
        changed_backends = ci_utils.BACKEND_ALIASES.values()

    full_name_to_alias = {v: k for k, v in ci_utils.BACKEND_ALIASES.items()}

    for item in items:
        path = item.location[0]

        for key, value in ci_utils.BACKEND_ALIASES.items():
            if f"/{key}/" in path:
                item.add_marker(getattr(pytest.mark, value))

        for marker in ci_utils.OTHER_MARKERS:
            if f"/{marker}/" in path:
                item.add_marker(getattr(pytest.mark, marker))

        for changed in changed_backends:
            if changed in full_name_to_alias:
                changed = full_name_to_alias[changed]
            if f"/{changed}/" in path:
                item.add_marker(pytest.mark.changed)

        for always in ci_utils.SHARED_MODULES:
            if always.startswith("/models/"):
                always = always[len("/models/") :]

            if f"/unit/{always}" in path:
                item.add_marker(pytest.mark.always)
                item.add_marker(pytest.mark.changed)
                for value in ci_utils.BACKEND_ALIASES.values():
                    item.add_marker(getattr(pytest.mark, value))
                for marker in ci_utils.OTHER_MARKERS:
                    item.add_marker(getattr(pytest.mark, marker))


def get_benchmark_info():
    uname = platform.uname()
    (commitHash, commitTime) = utils.getCommitInfo()

    return BenchmarkInfo(
        machineName=uname.machine,
        cudaVer="na",
        osType="%s %s" % (uname.system, uname.release),
        pythonVer=platform.python_version(),
        commitHash=commitHash,
        commitTime=commitTime,
        gpuType="na",
        cpuType=uname.processor,
        arch=uname.machine,
        ram="%d" % psutil.virtual_memory().total,
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup_dataloader():
    """After each test runs. Call .stop() on any dataloaders created during the test.
    The avoids issues with background threads hanging around and interfering with subsequent tests.
    This happens when a dataloader is partially consumed (not all batches are iterated through).
    """
    with patch.object(
        LoaderBase, "__iter__", side_effect=LoaderBase.__iter__, autospec=True
    ) as patched:
        yield
        for call in patched.call_args_list:
            call.args[0].stop()
