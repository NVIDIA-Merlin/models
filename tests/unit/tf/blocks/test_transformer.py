import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset

from transformers import BertConfig
from merlin.schema import Tags
from merlin.models.tf.utils import testing_utils


def test_transformer_encoder():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128

    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))
    transformer_encod = mm.TransformerBlock(transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16))
    outputs = transformer_encod(inputs)

    assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


def test_transformer_encoder_for_classification():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128

    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))
    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, 
        num_attention_heads=16),
        output_fn=lambda x: x.pooler_output
        )
    outputs = transformer_encod(inputs)

    assert list(outputs.shape) == [NUM_ROWS, EMBED_DIM]


def test_classfication_model(sequence_testing_data: Dataset):
    EMBED_DIM = 48
    schema = sequence_testing_data.schema.select_by_name(['item_id_seq', 'categories'])
    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, 
        num_attention_heads=16),
        output_fn=lambda x: x.pooler_output
    )
    model = mm.Model(
        mm.InputBlockV2(
            schema,
            embeddings=mm.Embeddings(
                schema,
                sequence_combiner=None
                ),
        ),
        transformer_encod,
        mm.CategoricalPrediction(prediction=sequence_testing_data.schema["user_country"])
    )

    batch = mm.sample_batch(
        sequence_testing_data, batch_size=100, include_targets=False, to_ragged=True
    )
    outputs = model(batch)
    assert list(outputs.shape) ==[100, 63]
    model, losses = testing_utils.model_test(model, sequence_testing_data)
