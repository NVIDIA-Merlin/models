import pytest
from testbook import testbook


@testbook("examples/usecases/social.ipynb", execute=True)
@pytest.mark.example
def test_tf_example_social_usecase(tb):
    assert tb.cell_output_text(2)[-20:] == "parameters: 42331522"
    assert tb.cell_output_text(4)[-20:] == "parameters: 41739655"
