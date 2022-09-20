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

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(REPO_ROOT / "examples/01-Getting-started.ipynb", execute=False)
def test_example_01_getting_started(tb):
    tb.inject(
        """
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
        mock_train, mock_valid = generate_data(
            input="movielens-1m",
            num_rows=1000,
            set_sizes=(0.8, 0.2)
        )
        p1 = patch(
            "merlin.datasets.entertainment.get_movielens",
            return_value=[mock_train, mock_valid]
        )
        p1.start()
        """
    )
    tb.execute()
    metrics = tb.ref("metrics")
    assert set(metrics.keys()) == set(
        [
            "auc",
            "binary_accuracy",
            "loss",
            "regularization_loss",
            "precision",
            "recall",
        ]
    )
