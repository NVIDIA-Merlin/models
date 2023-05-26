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
import argparse

from merlin.models.utils.ci_utils import COMPARE_BRANCH, backend_has_changed, get_changed_backends

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=False, help="specific backend to check for changes"
    )
    parser.add_argument(
        "--branch",
        type=str,
        required=False,
        default=COMPARE_BRANCH,
        help="specific backend to check for changes",
    )
    args = parser.parse_args()

    if args.backend:
        print(str(backend_has_changed(args.backend, args.branch)).lower())
    else:
        print(" ".join(get_changed_backends(args.branch)))
