import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.sampling.base import AddRandomNegativesToBatch


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
