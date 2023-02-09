import logging
import os

import numpy as np
import tensorflow as tf
from args_parsing import Task, parse_arguments
from mtl import get_mtl_loss_weights, get_mtl_prediction_tasks
from ranking_models import get_model
from run_logging import WandbLogger, get_callbacks

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.schema.tags import Tags


def get_datasets(args):
    train_ds = Dataset(os.path.join(args.train_path, "*.parquet"), part_size="500MB")
    eval_ds = Dataset(os.path.join(args.eval_path, "*.parquet"), part_size="500MB")

    return train_ds, eval_ds


class RankingTrainEvalRunner:

    logger = None
    train_ds = None
    eval_ds = None
    train_loader = None
    eval_loader = None
    args = None

    def __init__(self, logger, train_ds, eval_ds, args):
        self.args = args
        self.logger = logger
        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.schema, self.targets = self.filter_schema_with_selected_targets(self.train_ds.schema)
        self.set_dataloaders(self.schema)

    def get_targets(self, schema):
        tasks = self.args.tasks
        if tasks == "all":
            tasks = schema.select_by_tag(Tags.TARGET).column_names
        elif "," in tasks:
            tasks = tasks.split(",")
        else:
            tasks = [tasks]

        targets_schema = schema.select_by_name(tasks)
        targets = dict()

        binary_classif_targets = targets_schema.select_by_tag(
            Tags.BINARY_CLASSIFICATION
        ).column_names
        if len(binary_classif_targets) > 0:
            targets[Task.BINARY_CLASSIFICATION] = binary_classif_targets

        regression_targets = targets_schema.select_by_tag(Tags.REGRESSION).column_names
        if len(regression_targets) > 0:
            targets[Task.REGRESSION] = regression_targets

        return targets

    def filter_schema_with_selected_targets(self, schema):
        targets = self.get_targets(schema)
        flattened_targets = [y for x in targets.values() for y in x]

        if self.args.tasks != "all":
            # Removing targets not used from schema
            targets_to_remove = list(
                set(schema.select_by_tag(Tags.TARGET).column_names).difference(
                    set(flattened_targets)
                )
            )
            schema = schema.excluding_by_name(targets_to_remove)

        return schema, targets

    def set_dataloaders(self, schema):
        args = self.args
        train_loader_kwargs = {}
        if self.args.in_batch_negatives_train:
            train_loader_kwargs["transform"] = InBatchNegatives(
                schema, args.in_batch_negatives_train
            )
        self.train_loader = mm.Loader(
            self.train_ds,
            batch_size=args.train_batch_size,
            schema=schema,
            **train_loader_kwargs,
        )

        eval_loader_kwargs = {}
        if args.in_batch_negatives_eval:
            eval_loader_kwargs["transform"] = InBatchNegatives(schema, args.in_batch_negatives_eval)

        self.eval_loader = mm.Loader(
            self.eval_ds,
            batch_size=args.eval_batch_size,
            schema=schema,
            **eval_loader_kwargs,
        )

    def get_metrics(self):
        metrics = dict()
        if Task.BINARY_CLASSIFICATION in self.targets:
            metrics.update(
                {
                    f"{target}/binary_output": tf.keras.metrics.AUC(
                        name="auc", num_thresholds=int(1e5)
                    )
                    for target in self.targets[Task.BINARY_CLASSIFICATION]
                }
            )

        if Task.REGRESSION in self.targets:
            metrics.update(
                {f"{target}/regression_output": "rmse" for target in self.targets[Task.REGRESSION]}
            )

        if len(metrics) == 1:
            return list(metrics.values())[0]
        else:
            return metrics

    def get_optimizer(self):
        lerning_rate = self.args.lr
        if self.args.lr_decay_rate:
            lerning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lr,
                decay_steps=self.args.lr_decay_steps,
                decay_rate=self.args.lr_decay_rate,
                staircase=True,
            )

        if self.args.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                learning_rate=lerning_rate,
            )
        elif self.args.optimizer == "adagrad":
            opt = tf.keras.optimizers.Adagrad(
                learning_rate=lerning_rate,
            )
        else:
            raise ValueError("Invalid optimizer")

        return opt

    def train_eval_stl(self):
        if Task.BINARY_CLASSIFICATION in self.targets:
            prediction_task = mm.BinaryOutput(self.targets[Task.BINARY_CLASSIFICATION][0])
        elif Task.REGRESSION in self.targets:
            prediction_task = mm.RegressionOutput(self.targets[Task.REGRESSION][0])
        else:
            raise ValueError(f"Unrecognized task: {self.targets}")

        model = get_model(self.schema, prediction_task, self.args)

        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
        )

        callbacks = get_callbacks(self.args)
        class_weights = {0: 1.0, 1: self.args.stl_positive_class_weight}

        logging.info("Starting to train the model")
        model.fit(
            self.train_loader,
            epochs=self.args.epochs,
            batch_size=self.args.train_batch_size,
            steps_per_epoch=self.args.train_steps_per_epoch,
            shuffle=False,
            drop_last=False,
            callbacks=callbacks,
            # validation_data=self.eval_ds,
            # validation_steps=self.args.validation_steps,
            train_metrics_steps=self.args.train_metrics_steps,
            class_weight=class_weights,
        )
        logging.info("Starting the evaluation of the model")

        eval_metrics = model.evaluate(
            self.eval_loader,
            batch_size=self.args.eval_batch_size,
            return_dict=True,
            callbacks=callbacks,
        )

        print(f"EVALUATION METRICS: {eval_metrics}")
        self.log_final_metrics(eval_metrics)

        return model

    def train_eval_mtl(self):
        args = self.args

        prediction_tasks = get_mtl_prediction_tasks(self.targets, self.args)

        model = get_model(self.schema, prediction_tasks, self.args)

        loss_weights = get_mtl_loss_weights(args, self.targets)

        metrics = self.get_metrics()
        model.compile(
            self.get_optimizer(),
            run_eagerly=False,
            metrics=metrics,
            loss_weights=loss_weights,
        )
        callbacks = get_callbacks(self.args)

        logging.info(f"MODEL: {model}")

        logging.info("Starting to train the model (fit())")
        model.fit(
            self.train_loader,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            steps_per_epoch=args.train_steps_per_epoch,
            shuffle=False,
            drop_last=False,
            callbacks=callbacks,
            # validation_data=valid_ds,
            # validation_steps=args.validation_steps,
            train_metrics_steps=args.train_metrics_steps,
        )

        logging.info("Starting the evaluation the model (evaluate())")

        eval_metrics = model.evaluate(
            self.eval_loader,
            batch_size=args.eval_batch_size,
            return_dict=True,
            callbacks=callbacks,
        )

        auc_metric_results = {
            k.split("/")[0]: v
            for k, v in eval_metrics.items()
            if "binary_classification_task_auc" in k
        }

        auc_metric_results = {f"{k}-auc": v for k, v in auc_metric_results.items()}

        avg_metrics = {
            "auc_avg": np.mean(list(auc_metric_results.values())),
        }

        all_metrics = {
            **avg_metrics,
            **auc_metric_results,
            **eval_metrics,
        }

        logging.info(f"EVALUATION METRICS: {all_metrics}")

        # log final metrics
        self.log_final_metrics(all_metrics)

        return model

    def log_final_metrics(self, metrics_results):
        if self.logger:
            metrics_results = {f"{k}-final": v for k, v in metrics_results.items()}
            self.logger.log(metrics_results)

    def run(self):
        if self.logger:
            self.logger.setup()

        tf.keras.utils.set_random_seed(self.args.random_seed)

        try:
            logging.info(f"TARGETS: {self.targets}")

            if len(self.targets) == 1 and len(list(self.targets.values())[0]) == 1:
                # Single target = Single-Task Learning
                model = self.train_eval_stl()
            else:
                # Multiple targets = Multi-Task Learning
                model = self.train_eval_mtl()

            logging.info("Finished training and evaluation")

            if self.args.save_trained_model_path:
                logging.info(f"Saving model to {self.args.save_trained_model_path}")
                model.save(self.args.save_trained_model_path)

            logging.info("Script successfully finished")

        finally:
            if self.logger:
                self.logger.teardown()


def main():
    args = parse_arguments()

    logger = None
    if args.log_to_wandb:
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, config=args)

    train_ds, eval_ds = get_datasets(args)

    runner = RankingTrainEvalRunner(logger, train_ds, eval_ds, args)
    runner.run()


if __name__ == "__main__":
    main()
