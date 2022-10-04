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

import importlib
import json
import os
import pathlib

from merlin.models.api import MerlinModel


def load_model(path: os.PathLike) -> MerlinModel:
    """
    Load a model from path where Merlin Model is saved.
    """
    load_path = pathlib.Path(path)

    if not load_path.is_dir():
        raise ValueError("path provided to 'load' must be a directory.")

    model_metadata_path = load_path / "merlin_metadata" / "model_metadata.json"
    with open(model_metadata_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    model_module_name = model_metadata["model_module_name"]
    model_class_name = model_metadata["model_class_name"]
    model_module = importlib.import_module(model_module_name)
    model_cls = getattr(model_module, model_class_name)
    return model_cls.load(path)
