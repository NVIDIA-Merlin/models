import numpy as np
import pytest
import tensorflow as tf
from transformers import BertConfig

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.loader import Loader
from merlin.models.tf.transformers.block import (
    AlbertBlock,
    BertBlock,
    GPT2Block,
    RobertaBlock,
    XLNetBlock,
)
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_import():
    import transformers

    assert transformers is not None


def test_transformer_encoder():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128

    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))
    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16)
    )
    outputs = transformer_encod(inputs)

    assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


def test_transformer_encoder_with_pooling():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128

    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))
    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16),
        transformer_post="pooler_output",
    )
    outputs = transformer_encod(inputs)

    assert list(outputs.shape) == [NUM_ROWS, EMBED_DIM]


@pytest.mark.parametrize("max_seq_length", [None, 5, 20])
def test_transformer_encoder_with_list_to_dense(max_seq_length):
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128
    inputs = tf.RaggedTensor.from_tensor(tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM)))

    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16),
        pre=mm.ListToDense(max_seq_length=max_seq_length),
    )
    outputs = transformer_encod(inputs)

    if max_seq_length is not None:
        assert list(outputs.shape) == [NUM_ROWS, max_seq_length, EMBED_DIM]
    else:
        assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


@pytest.mark.parametrize("encoder", [XLNetBlock, BertBlock, AlbertBlock, RobertaBlock, GPT2Block])
def test_hf_tranformers_blocks(encoder):
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128
    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))
    transformer_encod = encoder(
        d_model=EMBED_DIM,
        n_head=8,
        n_layer=2,
    )
    outputs = transformer_encod(inputs)
    assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_as_classfication_model(sequence_testing_data: Dataset, run_eagerly):
    EMBED_DIM = 48
    loader, schema = classification_loader(sequence_testing_data)

    model = mm.Model(
        mm.InputBlockV2(
            schema,
            embeddings=mm.Embeddings(schema, sequence_combiner=None),
        ),
        BertBlock(
            d_model=EMBED_DIM,
            n_head=8,
            n_layer=2,
            transformer_post="pooler_output",
        ),
        mm.CategoricalOutput(
            to_call=schema["user_country"],
        ),
    )

    batch = next(iter(loader))[0]
    outputs = model(batch)
    assert list(outputs.shape) == [50, 63]
    testing_utils.model_test(model, loader, run_eagerly=run_eagerly)


def test_tranformer_with_prepare_module(sequence_testing_data):
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128
    inputs = tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM))

    class DummyPrepare(tf.keras.layers.Layer):
        def __init__(self, transformer, **kwargs):
            self.transformer = transformer
            super().__init__(**kwargs)

        def call(self, inputs, features=None):
            bs = tf.shape(inputs)[0]
            seq_len = self.transformer.config.max_position_embeddings
            attention_mask = tf.ones((bs, seq_len))
            inputs = {"inputs_embeds": inputs, "attention_mask": attention_mask}
            return inputs

    transformer_encod = BertBlock(
        d_model=EMBED_DIM,
        n_head=8,
        n_layer=2,
        max_position_embeddings=SEQ_LENGTH,
        transformer_pre=DummyPrepare,
    )

    outputs = transformer_encod(inputs)
    assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


def classification_loader(sequence_testing_data: Dataset):
    def _target_to_onehot(inputs, targets):
        targets = tf.squeeze(tf.one_hot(targets, 63))
        return inputs, targets

    schema = sequence_testing_data.schema.select_by_name(
        ["item_id_seq", "categories", "user_country"]
    )
    schema["user_country"] = schema["user_country"].with_tags(
        schema["user_country"].tags + "target"
    )
    sequence_testing_data.schema = schema
    dataloader = mm.Loader(sequence_testing_data, batch_size=50, transform=_target_to_onehot)
    return dataloader, schema


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_causal_language_modeling(sequence_testing_data: Dataset, run_eagerly):

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(
        Tags.CATEGORICAL
    )
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictNext(schema=seq_schema, target=target)

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False, transform=predict_next)

    model = mm.Model(
        mm.InputBlockV2(
            seq_schema,
            embeddings=mm.Embeddings(
                seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        GPT2Block(d_model=48, n_head=8, n_layer=2),
        mm.CategoricalOutput(
            seq_schema.select_by_name(target), default_loss="categorical_crossentropy"
        ),
    )

    batch = next(iter(loader))[0]
    outputs = model(batch)
    assert list(outputs.shape) == [8, 3, 51997]
    testing_utils.model_test(model, loader, run_eagerly=run_eagerly, reload_model=True)

    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True)
    assert len(metrics) > 0

    predictions = model.predict(loader, batch_size=8, steps=1)
    assert predictions.shape == (8, 3, 51997)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_masked_language_modeling(sequence_testing_data: Dataset, run_eagerly):

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(
        Tags.CATEGORICAL
    )
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)
    model = mm.Model(
        mm.InputBlockV2(
            seq_schema,
            embeddings=mm.Embeddings(
                seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        # BertBlock(d_model=48, n_head=8, n_layer=2, pre=mm.ReplaceMaskedEmbeddings()),
        GPT2Block(d_model=48, n_head=4, n_layer=2, pre=mm.ReplaceMaskedEmbeddings()),
        mm.CategoricalOutput(
            seq_schema.select_by_name(target),
            default_loss="categorical_crossentropy",
        ),
    )
    seq_mask_random = mm.SequenceMaskRandom(schema=seq_schema, target=target, masking_prob=0.3)

    inputs, targets = next(iter(loader))
    outputs = model(inputs, targets=targets, training=True)
    assert list(outputs.shape) == [8, 4, 51997]
    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=True,
        fit_kwargs={"pre": seq_mask_random},
    )

    seq_mask_last = mm.SequenceMaskLast(schema=seq_schema, target=target)
    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True, pre=seq_mask_last)
    assert len(metrics) > 0

    predictions = model.predict(loader, batch_size=8, steps=1)
    # TODO: Decide what should be the output of predictions for MLM (currently it predicts for all
    # positions of the sequence, but typically you want a single next-item prediction)
    assert predictions.shape == (8, 4, 51997)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_masked_language_modeling_check_eval_masked(
    sequence_testing_data: Dataset, run_eagerly
):

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(
        Tags.CATEGORICAL
    )
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)
    model = mm.Model(
        mm.InputBlockV2(
            seq_schema,
            embeddings=mm.Embeddings(
                seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        # BertBlock(d_model=48, n_head=8, n_layer=2, pre=mm.ReplaceMaskedEmbeddings()),
        GPT2Block(d_model=48, n_head=4, n_layer=2, pre=mm.ReplaceMaskedEmbeddings()),
        mm.CategoricalOutput(
            seq_schema.select_by_name(target),
            default_loss="categorical_crossentropy",
        ),
    )
    seq_mask_random = mm.SequenceMaskRandom(schema=seq_schema, target=target, masking_prob=0.3)

    inputs, targets = next(iter(loader))
    outputs = model(inputs, targets=targets, training=True)
    assert list(outputs.shape) == [8, 4, 51997]
    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=True,
        fit_kwargs={"pre": seq_mask_random},
        metrics=[mm.RecallAt(5000), mm.NDCGAt(5000)],
    )

    # This transform only extracts targets, but without applying mask
    seq_target_as_input_no_mask = mm.SequenceTargetAsInput(schema=seq_schema, target=target)
    metrics_all_positions1 = model.evaluate(
        loader, batch_size=8, steps=1, return_dict=True, pre=seq_target_as_input_no_mask
    )
    metrics_all_positions2 = model.evaluate(
        loader, batch_size=8, steps=1, return_dict=True, pre=seq_target_as_input_no_mask
    )

    def _metrics_almost_equal(metrics1, metrics2):
        return np.all(
            [
                np.isclose(metrics1[k], metrics2[k])
                for k in metrics1
                if k not in "regularization_loss"
            ]
        )

    # Ensures metrics without masked positions are equal
    assert _metrics_almost_equal(metrics_all_positions1, metrics_all_positions2)

    seq_mask_last = mm.SequenceMaskLast(schema=seq_schema, target=target)
    metrics_last_positions = model.evaluate(
        loader, batch_size=8, steps=1, return_dict=True, pre=seq_mask_last
    )
    # Ensures metrics masking only last positions are different then the ones
    # considering all positions
    assert not _metrics_almost_equal(metrics_all_positions1, metrics_last_positions)
