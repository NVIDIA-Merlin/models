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

from pathlib import Path
from typing import Set

import distributed
import pytest

from merlin.core.utils import Distributed
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset

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


BACKEND_ALIASES = {
    "tf": "tensorflow",
    "torch": "torch",
    "implicit": "implicit",
    "lightfm": "lightfm",
    "xgb": "xgboost",
}

OTHER_MARKERS = {"unit", "integration", "datasets", "horovod", "transformers"}

SHARED = {
    "/datasets/",
    "/models/config/",
    "/models/utils/",
    "/models/io.py",
}


def get_changed_backends() -> Set[str]:
    try:
        from git import Repo

        repo = Repo()

        changed_files = {item.a_path for item in repo.index.diff("HEAD")}
        untracked_files = {file for file in repo.untracked_files}

        # Use the git diff command to get unstaged changes
        unstaged_files = repo.git.diff(None, name_only=True).split()

        all_changed_files = changed_files.union(untracked_files, unstaged_files)

        changed_backends = set()
        for file in all_changed_files:
            try:
                # If shared file is updated, we need to run all backends
                for shared in SHARED:
                    if shared in file:
                        return set(BACKEND_ALIASES.keys())

                name = file.split("/")[2]
                if name in BACKEND_ALIASES:
                    changed_backends.add(name)
            except IndexError:
                continue

        return changed_backends
    except ImportError:
        return set()


def pytest_collection_modifyitems(items):
    changed_backends = get_changed_backends()

    for item in items:
        path = item.location[0]

        for key, value in BACKEND_ALIASES.items():
            if f"/{key}/" in path:
                item.add_marker(getattr(pytest.mark, value))

        for marker in OTHER_MARKERS:
            if f"/{marker}/" in path:
                item.add_marker(getattr(pytest.mark, marker))

        for changed in changed_backends:
            if f"/{changed}/" in path:
                item.add_marker(pytest.mark.changed)
            else:
                item.add_marker(pytest.mark.unchanged)

        for always in SHARED:
            if always.startswith("/models/"):
                always = always[len("/models/") :]

            if "/unit/" + always in path:
                item.add_marker(pytest.mark.changed)
                for value in BACKEND_ALIASES.values():
                    item.add_marker(getattr(pytest.mark, value))
                for marker in OTHER_MARKERS:
                    item.add_marker(getattr(pytest.mark, marker))
