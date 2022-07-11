import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.blocks.core.combinators import PerMode
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.sampling.base import AddRandomNegativesToBatch
from merlin.models.tf.utils import testing_utils


def test_negatives_to_batch(music_streaming_data: Dataset):
    schema = music_streaming_data.schema
    batch_size, n_per_positive = 10, 5
    features = mm.sample_batch(
        music_streaming_data, batch_size=batch_size, include_targets=False, to_dense=True
    )

    sampler = AddRandomNegativesToBatch(schema, 5)
    with_negatives = sampler(features)

    expected_batch_size = batch_size + batch_size * n_per_positive
    assert all(f.shape[0] == expected_batch_size for f in with_negatives.values())


def test_negatives_to_batch_in_input_block(music_streaming_data: Dataset):
    add_negatives = AddRandomNegativesToBatch(music_streaming_data.schema, 5)
    negative_sampling = PerMode(training=add_negatives)

    model = mm.Model(
        mm.InputBlock(music_streaming_data.schema, post=negative_sampling),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
    )

    batch_size, n_per_positive = 10, 5
    features = mm.sample_batch(
        music_streaming_data, batch_size=batch_size, include_targets=False, to_dense=True
    )

    expected_batch_size = batch_size + batch_size * n_per_positive
    with_negatives = model(features, training=True)
    assert with_negatives.shape[0] == expected_batch_size

    without_negatives = model(features)
    assert without_negatives.shape[0] == batch_size


def test_negatives_to_batch_in_dataloader(music_streaming_data: Dataset):
    add_negatives = AddRandomNegativesToBatch(music_streaming_data.schema, 5)

    batch_size, n_per_positive = 10, 5
    dataset = BatchedDataset(music_streaming_data, batch_size=batch_size)
    dataset = dataset.map(add_negatives)

    features = next(iter(dataset))
    expected_batch_size = batch_size + batch_size * n_per_positive
    assert all(f.shape[0] == expected_batch_size for f in features.values())

    model = mm.Model(
        mm.InputBlock(music_streaming_data.schema),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
    )
    assert model(features).shape[0] == expected_batch_size

    testing_utils.model_test(model, dataset)
