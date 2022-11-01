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
import os
import pathlib
from typing import Optional, Union

from merlin.models.api import MerlinModel
from merlin.models.utils.schema_utils import schema_to_tensorflow_metadata_json
from merlin.schema import Schema

_MERLIN_METADATA_DIR_NAME = ".merlin"


def save_merlin_metadata(
    export_path: Union[str, os.PathLike],
    model: MerlinModel,
    input_schema: Optional[Schema],
    output_schema: Optional[Schema],
) -> None:
    """Saves data to Merlin Metadata Directory."""
    export_path = pathlib.Path(export_path)
    merlin_metadata_dir = export_path / _MERLIN_METADATA_DIR_NAME
    merlin_metadata_dir.mkdir(exist_ok=True)

    if input_schema is not None:
        schema_to_tensorflow_metadata_json(
            input_schema,
            merlin_metadata_dir / "input_schema.json",
        )
    if output_schema is not None:
        schema_to_tensorflow_metadata_json(
            output_schema,
            merlin_metadata_dir / "output_schema.json",
        )
