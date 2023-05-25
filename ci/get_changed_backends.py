import os
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

SHARED = {
    "/datasets/",
    "/models/config/",
    "/models/utils/",
    "/models/io.py",
}


def get_changed_backends() -> Set[str]:
    """Check which backends need to be tested based on the changed files in the current branch."""
    repo = Repo()

    # If on the main branch, everything is changed
    if os.environ.get("GITHUB_REF", "") == "refs/heads/main":
        return set(BACKEND_ALIASES.keys())

    commit = repo.head.commit  # Current branch last commit
    diffs = commit.diff(repo.commit("main"))

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
            for shared in SHARED:
                if shared in file:
                    return BACKEND_ALIASES.keys()

            name = file.split("/")[2]
            if name in BACKEND_ALIASES:
                changed_backends.add(name)
        except IndexError:
            continue

    return changed_backends


if __name__ == "__main__":
    print(" ".join(get_changed_backends()))
