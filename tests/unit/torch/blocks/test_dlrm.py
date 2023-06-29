import merlin.models.torch as mm
from merlin.models.torch.batch import sample_batch
from merlin.models.torch.utils import module_utils
from merlin.schema import Tags


class TestDLRMInputBlock:
    ...


class DLRMInteraction:
    ...


class TestDLRMBlock:
    def test_basic(self, testing_data):
        schema = testing_data.schema
        embedding_dim = 64
        block = mm.DLRMBlock(
            schema,
            dim=embedding_dim,
            bottom_block=mm.MLPBlock([embedding_dim]),
        )
        batch_size = 16
        batch = sample_batch(testing_data, batch_size=batch_size)
        block.to(device=batch.device())  # TODO: move this

        outputs = module_utils.module_test(block, batch.features)

        num_features = len(schema.select_by_tag(Tags.CATEGORICAL)) + 1
        dot_product_dim = (num_features - 1) * num_features // 2
        assert list(outputs.shape) == [batch_size, dot_product_dim + embedding_dim]
