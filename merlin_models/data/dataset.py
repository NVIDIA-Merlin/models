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


import os
from typing import Optional

from merlin_standard_lib import Schema

from .synthetic import generate_user_item_interactions


class Dataset:
    def __init__(self, schema_path: str, device: str = "cpu"):
        self.schema_path = schema_path
        if self.schema_path.endswith(".pb") or self.schema_path.endswith(".pbtxt"):
            self._schema = Schema().from_proto_text(self.schema_path)
        else:
            self._schema = Schema().from_json(self.schema_path)
        self.device = device

    @property
    def schema(self) -> Schema:
        return self._schema

    def generate_synthetic_interactions(
        self, num_rows=100, min_session_length=5, max_session_length=None, save_path=None
    ):
        data = generate_user_item_interactions(
            num_rows, self.schema, min_session_length, max_session_length, self.device
        )
        if save_path:
            data.to_parquet(save_path)
        return data

    def tf_synthetic_tensors(
        self,
        num_rows=100,
        min_session_length=5,
        max_session_length=None,
    ):
        import tensorflow as tf

        data = self.generate_synthetic_interactions(
            num_rows, min_session_length, max_session_length
        )
        if self.device == "gpu":
            data = data.to_pandas()
        data = data.to_dict("list")
        return {key: tf.convert_to_tensor(value) for key, value in data.items()}

    def torch_synthetic_tensors(
        self,
        num_rows=100,
        min_session_length=5,
        max_session_length=None,
    ):
        import torch

        data = self.generate_synthetic_interactions(
            num_rows, min_session_length, max_session_length
        )
        if self.device == "gpu":
            data = data.to_pandas()
        data = data.to_dict("list")
        return {key: torch.tensor(value).to(self.device) for key, value in data.items()}


class ParquetDataset(Dataset):
    def __init__(
        self,
        dir,
        parquet_file_name="data.parquet",
        schema_file_name="schema.json",
        schema_path: Optional[str] = None,
        device="cpu",
    ):
        super(ParquetDataset, self).__init__(
            schema_path or os.path.join(dir, schema_file_name), device=device
        )
        self.path = os.path.join(dir, parquet_file_name)

    def get_tf_dataloader(self):
        """return tf NVTabular loader"""
        raise NotImplementedError

    def get_torch_dataloader(self):
        """return torch NVTabular loader"""
        raise NotImplementedError
