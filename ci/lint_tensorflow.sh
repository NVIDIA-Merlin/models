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

#!/bin/bash

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__root="$(cd "$(dirname "${__dir}")" && pwd)"


# Check for use of tensorflow.python.keras

grep_output=$(grep -r "python.keras" --include "*.py" "${__root}/merlin/" "${__root}/tests/")

if [[ -n $grep_output ]]
then
    cat << EOF
FAIL: Found files with python.keras imports.
`tensorflow.keras.python` should not be used because it is incompatible with `tensorflow.keras`.
Instead tensorflow.keras is recommended.
EOF
    printf "%s\n" "${grep_output}"
    exit 1
else
    echo "OK: no files with python.keras found"
fi
