import os

import pytest

from merlin.models.data.ecommerce.aliccp import prepare_alliccp

# This is the path to the raw ali-ccp dataset
MAYBE_ALICCP_DATA = os.environ.get("DATA_PATH_ALICCP", None)


@pytest.mark.skipif(
    MAYBE_ALICCP_DATA is None,
    reason="ALI-CCP data is not available, pass it through env variable $DATA_PATH_ALICCP",
)
def test_prepare_alliccp(tmp_path):
    data_path = MAYBE_ALICCP_DATA

    prepare_alliccp(data_path, file_size=50, max_num_rows=100, output_dir=tmp_path)
    output_files = list(tmp_path.glob("*/*"))

    assert len(output_files) == 2
    assert all(f.name.endswith(".parquet") for f in output_files)
