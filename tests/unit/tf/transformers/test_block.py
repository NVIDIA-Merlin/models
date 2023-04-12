import itertools

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
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


@pytest.mark.parametrize("run_eagerly", [True])
def test_retrieval_transformer(sequence_testing_data: Dataset, run_eagerly):
    set_random_seed(42)

    sequence_testing_data.schema = sequence_testing_data.schema.select_by_tag(
        Tags.SEQUENCE
    ).select_by_tag(Tags.CATEGORICAL)
    seq_schema = sequence_testing_data.schema

    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_last = mm.SequencePredictLast(schema=seq_schema, target=target)
    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)

    query_schema = seq_schema
    output_schema = seq_schema.select_by_name(target)

    d_model = 48
    query_encoder = mm.Encoder(
        mm.InputBlockV2(
            query_schema,
            categorical=mm.Embeddings(
                query_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([d_model]),
        GPT2Block(d_model=d_model, n_head=2, n_layer=2),
        tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    )

    model = mm.RetrievalModelV2(
        query=query_encoder,
        output=mm.ContrastiveOutput(output_schema, negative_samplers="in-batch"),
    )

    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=False,
        metrics={},
        fit_kwargs={"pre": predict_last},
    )

    predictions = model.predict(loader)
    assert list(predictions.shape) == [100, 51997]

    query_embeddings = query_encoder.predict(loader)
    assert list(query_embeddings.shape) == [100, d_model]

    item_embeddings = model.candidate_embeddings().compute().to_numpy()

    assert list(item_embeddings.shape) == [51997, d_model]
    predicitons_2 = np.dot(query_embeddings, item_embeddings.T)

    np.testing.assert_allclose(predictions, predicitons_2, atol=1e-6)


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


def test_transformer_encoder_with_list_to_dense():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128
    inputs = tf.RaggedTensor.from_tensor(tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM)))

    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16),
        pre=mm.ToDense(),
    )
    outputs = transformer_encod(inputs)

    assert list(outputs.shape) == [NUM_ROWS, SEQ_LENGTH, EMBED_DIM]


def test_transformer_encoder_with_post():
    NUM_ROWS = 100
    SEQ_LENGTH = 10
    EMBED_DIM = 128
    inputs = tf.RaggedTensor.from_tensor(tf.random.uniform((NUM_ROWS, SEQ_LENGTH, EMBED_DIM)))

    transformer_encod = mm.TransformerBlock(
        transformer=BertConfig(hidden_size=EMBED_DIM, num_attention_heads=16),
        pre=mm.ToDense(),
        post="sequence_mean",
    )
    outputs = transformer_encod(inputs)
    testing_utils.assert_serialization(transformer_encod)
    assert list(outputs.shape) == [NUM_ROWS, EMBED_DIM]


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
def test_transformer_as_classification_model(sequence_testing_data: Dataset, run_eagerly):
    EMBED_DIM = 48
    loader, schema = classification_loader(sequence_testing_data)

    model = mm.Model(
        mm.InputBlockV2(
            schema,
            categorical=mm.Embeddings(schema, sequence_combiner=None),
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

    batch = loader.peek()[0]

    outputs = model(batch)
    assert list(outputs.shape) == [64, 63]
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
    schema = sequence_testing_data.schema.select_by_name(
        ["item_id_seq", "categories", "user_country"]
    )
    sequence_testing_data.schema = schema
    dataloader = mm.Loader(
        sequence_testing_data,
        batch_size=64,
    ).map(mm.ToTarget(schema, "user_country", one_hot=True))
    return dataloader, dataloader.output_schema


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_predict_random(sequence_testing_data: Dataset, run_eagerly):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(
        Tags.CATEGORICAL
    )
    target_schema = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID)
    target = target_schema.column_names[0]

    sequence_testing_data.schema = seq_schema + target_schema
    model_schema = sequence_testing_data.schema

    transformer_input_dim = 48
    transformer_block = GPT2Block(d_model=transformer_input_dim, n_head=8, n_layer=2)
    model = mm.Model(
        mm.InputBlockV2(
            model_schema,
            categorical=mm.Embeddings(
                model_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([transformer_input_dim]),
        transformer_block,
        mm.CategoricalOutput(
            model_schema.select_by_name(target), default_loss="categorical_crossentropy"
        ),
    )

    predict_next = mm.SequencePredictRandom(
        schema=seq_schema, target=target, transformer=transformer_block
    )
    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)

    testing_utils.model_test(
        model, loader, run_eagerly=run_eagerly, reload_model=True, fit_kwargs={"pre": predict_next}
    )

    predict_last = mm.SequencePredictLast(
        schema=seq_schema, target=target, transformer=transformer_block
    )
    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True, pre=predict_last)
    assert len(metrics) > 0


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_causal_language_modeling(sequence_testing_data: Dataset, run_eagerly):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(
        Tags.CATEGORICAL
    )
    target_schema = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID)
    target = target_schema.column_names[0]

    sequence_testing_data.schema = seq_schema + target_schema
    model_schema = sequence_testing_data.schema

    transformer_input_dim = 48
    transformer_block = GPT2Block(d_model=transformer_input_dim, n_head=8, n_layer=2)
    model = mm.Model(
        mm.InputBlockV2(
            model_schema,
            categorical=mm.Embeddings(
                model_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([transformer_input_dim]),
        transformer_block,
        mm.CategoricalOutput(
            model_schema.select_by_name(target), default_loss="categorical_crossentropy"
        ),
    )

    predict_next = mm.SequencePredictNext(
        schema=seq_schema, target=target, transformer=transformer_block
    )
    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)

    testing_utils.model_test(
        model, loader, run_eagerly=run_eagerly, reload_model=True, fit_kwargs={"pre": predict_next}
    )

    batch = next(iter(loader))[0]
    outputs = model(batch)
    assert list(outputs.shape) == [8, 51997]

    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True, pre=predict_next)
    assert len(metrics) > 0

    predictions = model.predict(loader, batch_size=8, steps=1)
    assert predictions.shape == (8, 51997)

    predict_last = mm.SequencePredictLast(
        schema=seq_schema, target=target, transformer=transformer_block
    )
    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True, pre=predict_last)
    assert len(metrics) > 0


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_masked_language_modeling(sequence_testing_data: Dataset, run_eagerly):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)

    target_schema = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID)
    model_schema = seq_schema + target_schema
    target = target_schema.column_names[0]

    sequence_testing_data.schema = model_schema

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)
    transformer_input_dim = 48
    transformer_block = XLNetBlock(d_model=transformer_input_dim, n_head=8, n_layer=2)
    model = mm.Model(
        mm.InputBlockV2(
            seq_schema,
            categorical=mm.Embeddings(
                seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([transformer_input_dim]),
        transformer_block,
        mm.CategoricalOutput(
            seq_schema.select_by_name(target),
            default_loss="categorical_crossentropy",
        ),
    )
    seq_mask_random = mm.SequenceMaskRandom(
        schema=seq_schema, target=target, masking_prob=0.3, transformer=transformer_block
    )

    inputs, targets = loader.peek()

    outputs = model(inputs, targets=targets, training=True)
    assert list(outputs.shape) == [8, 4, 51997]
    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=True,
        fit_kwargs={"pre": seq_mask_random},
    )

    seq_mask_last = mm.SequenceMaskLast(
        schema=seq_schema, target=target, transformer=transformer_block
    )
    metrics = model.evaluate(loader, batch_size=8, steps=1, return_dict=True, pre=seq_mask_last)
    assert len(metrics) > 0

    # Get predictions for next-item position
    predictions = model.predict(loader, batch_size=8, steps=1)
    assert predictions.shape == (8, 51997)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_with_masked_language_modeling_check_eval_masked(
    sequence_testing_data: Dataset, run_eagerly
):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)

    target_schema = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID)
    target = target_schema.column_names[0]
    model_schema = seq_schema + target_schema
    sequence_testing_data.schema = model_schema

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)

    transformer_input_dim = 48
    transformer_block = BertBlock(d_model=transformer_input_dim, n_head=8, n_layer=2)
    model = mm.Model(
        mm.InputBlockV2(
            seq_schema,
            categorical=mm.Embeddings(
                seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([transformer_input_dim]),
        transformer_block,
        mm.CategoricalOutput(
            seq_schema.select_by_name(target),
            default_loss="categorical_crossentropy",
        ),
    )
    seq_mask_random = mm.SequenceMaskRandom(
        schema=seq_schema, target=target, masking_prob=0.3, transformer=transformer_block
    )

    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=True,
        fit_kwargs={"pre": seq_mask_random},
        metrics=[mm.RecallAt(5000), mm.NDCGAt(5000, seed=4)],
    )

    inputs = itertools.islice(iter(loader), 1)
    outputs = model.predict(inputs, pre=seq_mask_random)
    assert list(outputs.shape) == [8, 51997]

    # This transform only extracts targets, but without applying mask
    seq_target_as_input_no_mask = mm.SequenceTargetAsInput(schema=seq_schema, target=target)

    with Loader(sequence_testing_data, batch_size=8, shuffle=False) as loader:
        metrics_all_positions1 = model.evaluate(
            loader, batch_size=8, steps=1, return_dict=True, pre=seq_target_as_input_no_mask
        )

    with Loader(sequence_testing_data, batch_size=8, shuffle=False) as loader:
        metrics_all_positions2 = model.evaluate(
            loader, batch_size=8, steps=1, return_dict=True, pre=seq_target_as_input_no_mask
        )

    def _metrics_almost_equal(metrics1, metrics2):
        return np.all(
            [
                np.isclose(metrics1[k], metrics2[k], atol=1e-05)
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


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_transformer_model_with_masking_and_broadcast_to_sequence(
    sequence_testing_data: Dataset, run_eagerly: bool
):
    schema = sequence_testing_data.schema
    seq_schema = schema.select_by_name(["item_id_seq", "categories", "item_age_days_norm"])
    context_schema = schema.select_by_name(["user_country", "user_age"])
    sequence_testing_data.schema = seq_schema + context_schema

    target = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    item_id_name = schema.select_by_tag(Tags.ITEM_ID).first.properties["domain"]["name"]

    input_block = mm.InputBlockV2(
        sequence_testing_data.schema,
        embeddings=mm.Embeddings(
            seq_schema.select_by_tag(Tags.CATEGORICAL)
            + context_schema.select_by_tag(Tags.CATEGORICAL),
            sequence_combiner=None,
        ),
        post=mm.BroadcastToSequence(context_schema, seq_schema),
    )

    dmodel = 32
    mlp_block = mm.MLPBlock([128, dmodel], activation="relu")
    transformer_block = mm.GPT2Block(
        d_model=dmodel,
        n_head=4,
        n_layer=2,
    )

    dense_block = mm.SequentialBlock(input_block, mlp_block, transformer_block)

    mlp_block2 = mm.MLPBlock([128, dmodel], activation="relu")

    prediction_task = mm.CategoricalOutput(
        to_call=input_block["categorical"][item_id_name],
    )
    model = mm.Model(dense_block, mlp_block2, prediction_task)

    fit_pre = mm.SequenceMaskRandom(
        schema=seq_schema, target=target, masking_prob=0.3, transformer=transformer_block
    )
    testing_utils.model_test(
        model,
        sequence_testing_data,
        run_eagerly=run_eagerly,
        reload_model=False,
        fit_kwargs={"pre": fit_pre},
    )
