import pytest
from testbook import testbook


@pytest.mark.example
@testbook("examples/usecases/e-commerce.ipynb", execute=True)
def test_tf_example_ecommerce_usecase(tb):
    assert tb.cell_output_text(2)[-19:] == "parameters: 1112962"
    assert tb.cell_output_text(4)[-18:] == "parameters: 393444"
