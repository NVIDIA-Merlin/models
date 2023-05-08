import logging
import time
from typing import Any, Dict

from tensorflow.keras.callbacks import Callback


class WandbLogger:
    def __init__(
        self,
        wandb_project: str = None,
        wandb_entity: str = None,
        config: Dict[str, Any] = {},
        logging_path: str = None,
        auto_init: bool = False,
    ):
        """Class to manage logging to Weights&Biases (W&B)
        service, available at https://wandb.ai

        Parameters
        ----------
        wandb_project : str, optional
            Name of the W&B project to
            log the runs. If not provided it is logged
            to the user default W&B project", by default None
        wandb_entity : str, optional
            Name of the W&B entity (team name) to
            log the runs. If not provided it is logged
            to the user default W&B team", by default None
        config : Dict[str, Any], optional
            Dictionary with the run configuration / hyperparameters.
            By default {}.
        logging_path : str, optional
            Path where the wandb will log information
            locally. By default None, which logs to a local
            ".wandb" folder
        auto_init : bool, optional
            If True, init() method will be called automatically
            when the class is instantiated to configure wandb.
            By default False
        """
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self._config = config
        self.logging_path = logging_path
        self._initialized = False

        import wandb

        self._wandb_lib = wandb

        if auto_init:
            self.init()

    def init(self):
        """Initializes W&B setting the project, team,
        logging dir and hyperparamteres
        """
        self._wandb_lib.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config=self._config,
            dir=self.logging_path,
        )

    def _check_wandb_init(self):
        if self._wandb_lib.run is None:
            raise ValueError(
                "You need first to initialize W&B with "
                "WandbLogger.init() or wandb.init() directly"
            )

    def config(self, config: Dict[str, Any] = {}):
        """Updates wandb run configuration
        (hyperparameters)

        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dict, by default {}
        """
        self._check_wandb_init()
        self._wandb_lib.config.update(config)

    def log(self, metrics: Dict[str, Any]):
        """Logs the metric values to wandb

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics keys and values
        """
        self._check_wandb_init()
        self._wandb_lib.log(metrics)

    def get_callback(self, metrics_log_frequency: int = 10, **callback_kwargs):
        """Returns a wandb.keras.WandbCallback() instance.
        for usage with model.fit() and model.evaluate().

        Parameters
        ----------
        metrics_log_frequency : int, optional
            Number of steps on which the metrics are logged, by default 10.
            Logging metrics every step might reduce training and evaluation
            throughput.
        """
        callback = self._wandb_lib.keras.WandbCallback(
            log_batch_frequency=metrics_log_frequency, **callback_kwargs
        )
        return callback

    def teardown(self, exit_code: int = 0):
        """Finish wandb logging gracefully

        Parameters
        ----------
        exit_code : int, optional
            Exit code for this run, by default 0
        """
        if self._wandb_lib.run is not None:
            self._wandb_lib.finish(exit_code=exit_code)


class ExamplesPerSecondCallback(Callback):
    """ExamplesPerSecond callback.
    This callback records the average_examples_per_sec and
    current_examples_per_sec metrics during training.
    """

    def __init__(
        self, batch_size: int, every_n_steps: int = 1, logger=None, log_to_console: bool = False
    ):
        """Logs the average_examples_per_sec and
        current_examples_per_sec metrics during training.

        Parameters
        ----------
        batch_size : int
            Informs the batch size, as it is used for
            computing the throughput statistics
        every_n_steps : int, optional
            Computes the metrics every N steps, by default 1
        logger : bool, optional
            Logs the average_examples_per_sec and
            current_examples_per_sec metrics to a logger.
            Typically it is the WandbLogger
        log_to_console : bool, optional
            Log with logging.debug(), so that the statistics
            can be logged to console or file with logging library
        """
        self._batch_size = batch_size
        self._every_n_steps = every_n_steps
        self._logger = logger
        self._log_to_console = log_to_console
        super(ExamplesPerSecondCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self._first_batch = True
        self._epoch_steps = 0
        self._train_batches_average_examples_per_sec = []

    def on_train_end(self, logs=None):
        average_examples_per_sec = self.get_avg_examples_per_sec()
        self._train_batches_average_examples_per_sec.append(average_examples_per_sec)

    def get_train_batches_mean_of_avg_examples_per_sec(self):
        if len(self._train_batches_average_examples_per_sec) > 0:
            return sum(self._train_batches_average_examples_per_sec) / float(
                len(self._train_batches_average_examples_per_sec)
            )
        else:
            return 0.0

    def get_avg_examples_per_sec(self):
        current_time = time.time()
        average_examples_per_sec = self._batch_size * (
            self._epoch_steps / (current_time - self._epoch_start_time)
        )
        return average_examples_per_sec

    def on_train_batch_end(self, batch, logs=None):
        # Discards the first batch, as it is used to compile the
        # graph and affects the average
        if self._first_batch:
            self._epoch_steps = 0
            self._first_batch = False
            self._epoch_start_time = time.time()
            self._last_recorded_time = time.time()
            return

        """Log the examples_per_sec metric every_n_steps."""
        self._epoch_steps += 1
        current_time = time.time()

        if self._epoch_steps % self._every_n_steps == 0:
            average_examples_per_sec = self.get_avg_examples_per_sec()
            current_examples_per_sec = self._batch_size * (
                self._every_n_steps / (current_time - self._last_recorded_time)
            )

            if self._log_to_console:
                logging.debug(
                    f"[Examples/sec - Epoch step: {self._epoch_steps}] "
                    f"current: {current_examples_per_sec:.2f}, avg: {average_examples_per_sec:.2f}"
                )

            if self._logger:
                self._logger.log(
                    {
                        "current_examples_per_sec": current_examples_per_sec,
                        "average_examples_per_sec": average_examples_per_sec,
                    }
                )

            self._last_recorded_time = current_time  # Update last_recorded_time
