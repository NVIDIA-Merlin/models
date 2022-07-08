#!/usr/bin/env bash
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

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__root="$(cd "$(dirname "${__dir}")" && pwd)"


# Check for use of tensorflow.python.keras

grep_output=$(grep -n -r "python.keras" --include "*.py" "${__root}/merlin/" "${__root}/tests/")

if [[ -n $grep_output ]]
then
    msg=$(cat << EOF
FAIL: Found files with python.keras imports.
tensorflow.keras should be used instead to avoid compatibility issues.
EOF
       )
    # display error message
    echo -e "\e[01;31m${msg}\e[0m"
    # strip out full path to root
    output=$(echo $grep_output | sed -E "s@${__root}/@@g")
    # print matching files
    printf "\n%s\n" "${output}"
    exit 1
else
    echo "OK: no files with python.keras found"
fi
