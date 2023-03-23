from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.loader import sample_batch
from merlin.schema import Tags


class TestTabularInputBlock:
    def test_tabular_data(self, testing_data):
        schema = testing_data.schema.without("categories")

        data = sample_batch(testing_data, batch_size=10, shuffle=False, include_targets=False)
        device = list(data.values())[0].device
        input_block = TabularInputBlock(schema).to(device)

        assert input_block.schema.column_names == [
            "user_id",
            "item_id",
            "user_country",
            "item_age_days_norm",
            "event_hour_sin",
            "event_hour_cos",
            "event_weekday_sin",
            "event_weekday_cos",
            "user_age",
        ]
        assert input_block.select_by_tag(Tags.ITEM_ID).schema == schema.select_by_tag(Tags.ITEM_ID)
        assert input_block.select_by_name("event_hour_sin").schema == schema.select_by_name(
            "event_hour_sin"
        )

        inputs = input_block(data)

        assert inputs.shape == (10, 62)
