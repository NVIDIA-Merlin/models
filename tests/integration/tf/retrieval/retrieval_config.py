import fiddle as fdl

from tests.integration.tf.retrieval.retrieval_utils import (
    RetrievalTrainEvalRunner,
    get_callbacks,
    get_dataset,
    get_dual_encoder_model,
    get_item_frequencies,
    get_loss,
    get_metrics,
    get_optimizer,
    get_samplers,
    get_schema,
    get_youtube_dnn_model,
)


def config_retrieval_train_eval_runner(model_type, data_path):
    def make_datasets(schema):
        get_dataset_partial = fdl.Partial(get_dataset, schema)
        train_ds = get_dataset_partial(dataset="train")
        train_ds.data_path = schema.data_path
        eval_ds = get_dataset_partial(dataset="valid")
        eval_ds.data_path = schema.data_path
        return train_ds, eval_ds

    def make_model(schema, train_ds, model_type="two_tower"):
        if model_type == "youtubednn":
            model = fdl.Config(get_youtube_dnn_model, schema)
        else:
            samplers = fdl.Config(get_samplers, schema)
            items_frequencies = fdl.Config(get_item_frequencies, schema, train_ds)
            model = fdl.Config(get_dual_encoder_model, schema, samplers, items_frequencies)
        return model

    schema_cfg = fdl.Config(get_schema, data_path=data_path)
    train_ds_cfg, eval_ds_cfg = make_datasets(schema_cfg)
    model_cfg = make_model(schema_cfg, train_ds_cfg, model_type=model_type)
    optimizer = fdl.Config(get_optimizer)
    metrics = fdl.Config(get_metrics)
    loss = fdl.Config(get_loss)
    callbacks = fdl.Config(get_callbacks)

    runner_cfg = fdl.Config(
        RetrievalTrainEvalRunner,
        schema=schema_cfg,
        train_ds=train_ds_cfg,
        eval_ds=eval_ds_cfg,
        model=model_cfg,
        optimizer=optimizer,
        metrics=metrics,
        loss=loss,
        callbacks=callbacks,
    )
    return runner_cfg
