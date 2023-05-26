#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from typing import Set

from git import Repo

BACKEND_ALIASES = {
    "tf": "tensorflow",
    "torch": "torch",
    "implicit": "implicit",
    "lightfm": "lightfm",
    "xgb": "xgboost",
}
OTHER_MARKERS = {"unit", "integration", "example", "datasets", "horovod", "transformers"}
SHARED_MODULES = {
    "/datasets/",
    "/models/config/",
    "/models/utils/",
    "/models/io.py",
}


def get_changed_backends() -> Set[str]:
    """
    Check which backends need to be tested based on the changed files in the current branch.

    This function scans the current branch of a repository and identifies files that have changed.
    If a change is detected in a backend file, the corresponding backend alias is added to a set.
    If a change is detected in a shared file, all backend aliases are returned.

    Example usage::
        >>> get_changed_backends()
        {'backend1', 'backend2'}

    Returns
    -------
    set
        A set of backends that need to be tested. The backends are identified based on the
        changed files in the current branch of the repository.
    """

    repo = Repo()

    commit = repo.head.commit  # Current branch last commit

    if "main" in repo.branches:
        main_branch = repo.branches["main"]
    elif "refs/heads/main" in repo.refs:
        main_branch = repo.refs["refs/heads/main"]
    else:
        raise ValueError("Could not find main branch")
    diffs = commit.diff(main_branch)

    changed_files = set()
    for change_type in ["A", "D", "R", "M", "T"]:
        for diff in diffs.iter_change_type(change_type):
            if diff.a_path:
                changed_files.add(diff.a_path)
            if diff.b_path:
                changed_files.add(diff.b_path)

    untracked_files = {file for file in repo.untracked_files}

    # Use the git diff command to get unstaged changes
    unstaged_files = repo.git.diff(None, name_only=True).split()

    all_changed_files = changed_files.union(untracked_files, unstaged_files)

    changed_backends = set()
    for file in all_changed_files:
        try:
            # If shared file is updated, we need to run all backends
            for shared in SHARED_MODULES:
                if shared in file:
                    return BACKEND_ALIASES.keys()

            name = file.split("/")[2]
            if name in BACKEND_ALIASES:
                changed_backends.add(name)
        except IndexError:
            continue

    return changed_backends


def get_default_branch(repo):
    """Get the default branch of the repository."""
    for remote in repo.remotes:
        for ref in remote.refs:
            if ref.remote_head == repo.head.ref.remote_head:
                return ref


def backend_has_changed(backend_name: str) -> bool:
    """
    Check if a specific backend needs to be tested based on the changed files in the current branch.

    This function utilizes the get_changed_backends function to check if a specific backend has
    experienced changes and hence requires testing.

    Parameters
    ----------
    backend_name : str
        The name of the backend to check for changes.

    Returns
    -------
    bool
        Returns True if the backend has changed and needs testing, False otherwise.

    """
    changed_backends = get_changed_backends()
    output: bool = False

    for backend in backend_name.split("|"):
        output |= backend in changed_backends

    return output
