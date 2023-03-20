from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.loader import sample_batch


class TestTabularInputBlock:
    def test_tabular_data(self, testing_data):
        schema = testing_data.schema.without("categories")

        data = sample_batch(testing_data, batch_size=10, shuffle=False, include_targets=False)
        device = list(data.values())[0].device
        input_block = TabularInputBlock(schema).to(device)

        inputs = input_block(data)

        assert inputs.shape == (10, 62)
