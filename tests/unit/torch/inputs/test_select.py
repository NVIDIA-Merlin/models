#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import pytest
import torch
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.batch import Batch, sample_batch
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class TestSelectKeys:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.batch: Batch = sample_batch(music_streaming_data, batch_size=10)
        self.schema: Schema = music_streaming_data.schema
        self.user_schema: Schema = mm.schema.select(self.schema, Tags.USER)

    def test_forward(self):
        select_user = mm.SelectKeys(self.user_schema)
        outputs = module_utils.module_test(select_user, self.batch.features)

        assert select_user.schema == self.user_schema
        assert select_user.extra_repr() == ", ".join(self.user_schema.column_names)

        for col in {"user_id", "country", "user_age"}:
            assert col in outputs

        assert "user_genres__values" in outputs
        assert "user_genres__offsets" in outputs
        assert select_user != nn.LayerNorm(10)

    def test_select(self):
        select_user = mm.SelectKeys(self.user_schema)

        user_id = Schema([self.user_schema["user_id"]])
        assert select_user.select(ColumnSchema("user_id")).schema == user_id
        assert select_user.select(Tags.USER).schema == self.user_schema

    def test_initialize_from_schema(self):
        select_user = mm.SelectKeys()
        select_user.initialize_from_schema(self.user_schema[["user_id"]])
        assert select_user.schema == self.user_schema[["user_id"]]


class TestSelectFeatures:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.batch: Batch = sample_batch(music_streaming_data, batch_size=10)
        self.schema: Schema = music_streaming_data.schema
        self.user_schema: Schema = mm.schema.select(self.schema, Tags.USER)

    def test_forward(self):
        selected = mm.SelectFeatures(self.user_schema)
        block = mm.Block(nn.Identity(), selected)
        assert selected.select(Tags.USER).select_keys == selected.select_keys

        outputs = mm.schema.trace(block, self.batch.features["session_id"], batch=self.batch)
        assert len(outputs) == 5
        assert mm.input_schema(block).column_names == ["input"]
        assert mm.feature_schema(block).column_names == [
            "user_id",
            "country",
            "user_age",
            "user_genres",
        ]

    def test_embeddings(self):
        schema = Schema(
            [
                ColumnSchema("user_embedding", tags=[Tags.EMBEDDING, Tags.USER], dims=(10,)),
            ]
        )

        selected = mm.SelectFeatures(schema)
        outputs = selected(self.batch, batch=mm.Batch({"user_embedding": torch.randn(10, 10)}))

        assert "user" in outputs
        assert outputs["user"].shape == (10, 10)
