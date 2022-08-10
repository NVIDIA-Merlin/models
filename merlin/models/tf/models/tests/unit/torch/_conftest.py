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

import numpy as np
import pandas as pd
import pytest
import torch

import merlin.models.torch as ml
from merlin.datasets.synthetic import generate_data

NUM_EXAMPLES = 1000
MAX_CARDINALITY = 100


# TODO: Remove this


@pytest.fixture
def tabular_schema():
    return generate_data("testing", 10).schema


@pytest.fixture
def torch_con_features():
    features = {}
    keys = [f"con_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = torch.rand((NUM_EXAMPLES, 1))

    return features


@pytest.fixture
def torch_cat_features():
    features = {}
    keys = [f"cat_{f}" for f in "abcdef"]

    for key in keys:
        features[key] = torch.randint(MAX_CARDINALITY, (NUM_EXAMPLES,))

    return features


@pytest.fixture
def torch_masking_inputs():
    # fixed parameters for tests
    NUM_EXAMPLES = 20
    MAX_LEN = 10
    PAD_TOKEN = 0
    hidden_dim = 16
    features = {}
    # generate random tensors for test
    features["input_tensor"] = torch.tensor(
        np.random.uniform(0, 1, (NUM_EXAMPLES, MAX_LEN, hidden_dim))
    )
    # create sequences
    labels = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (NUM_EXAMPLES, MAX_LEN)))
    # replace last 2 items by zeros to mimic padding
    labels[:, MAX_LEN - 2 :] = 0
    features["labels"] = labels
    features["padding_idx"] = PAD_TOKEN
    features["vocab_size"] = MAX_CARDINALITY

    return features


@pytest.fixture
def torch_seq_prediction_head_inputs():
    ITEM_DIM = 128
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = torch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
    features["vocab_size"] = MAX_CARDINALITY
    features["item_dim"] = ITEM_DIM
    return features


@pytest.fixture
def torch_ranking_metrics_inputs():
    POS_EXAMPLE = 30
    VOCAB_SIZE = 40
    features = {}
    features["scores"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, VOCAB_SIZE)))
    features["ks"] = torch.LongTensor([1, 2, 3, 5, 10, 20])
    features["labels_one_hot"] = torch.LongTensor(
        np.random.choice(a=[0, 1], size=(POS_EXAMPLE, VOCAB_SIZE))
    )

    features["labels"] = torch.tensor(np.random.randint(1, VOCAB_SIZE, (POS_EXAMPLE,)))
    return features


@pytest.fixture
def torch_seq_prediction_head_link_to_block():
    ITEM_DIM = 64
    POS_EXAMPLE = 25
    features = {}
    features["seq_model_output"] = torch.tensor(np.random.uniform(0, 1, (POS_EXAMPLE, ITEM_DIM)))
    features["item_embedding_table"] = torch.nn.Embedding(MAX_CARDINALITY, ITEM_DIM)
    features["labels_all"] = torch.tensor(np.random.randint(1, MAX_CARDINALITY, (POS_EXAMPLE,)))
    features["vocab_size"] = MAX_CARDINALITY
    features["item_dim"] = ITEM_DIM
    features["config"] = {
        "item": {
            "dtype": "categorical",
            "cardinality": MAX_CARDINALITY,
            "tags": ["categorical", "item"],
            "log_as_metadata": True,
        }
    }

    return features


@pytest.fixture
def torch_tabular_features(tabular_schema):
    return ml.TabularFeatures.from_schema(
        tabular_schema,
        max_sequence_length=20,
        continuous_projection=64,
        aggregation="concat",
    )


@pytest.fixture
def torch_tabular_data():
    dataset = generate_data("testing", num_rows=100)

    df = dataset.to_ddf().compute()
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    data = df.to_dict("list")

    return {key: torch.tensor(value) for key, value in data.items()}
