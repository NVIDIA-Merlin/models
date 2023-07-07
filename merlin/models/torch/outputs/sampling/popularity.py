from typing import Optional

import torch
from torch import nn

from merlin.models.torch.block import registry


class LogUniformSampler(object):
    def __init__(self, range_max: int, n_sample: int):
        """LogUniformSampler samples negative samples based on a log-uniform distribution.

        Credits to:
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/log_uniform_sampler.py

        TensorFlow Reference:
        https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`
        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))
        Our implementation fixes num_tries at 2 * n_sample,
        and the actual #samples will vary from run to run

        Parameters
        ----------
        range_max : int
            The maximum value of the range for the log-uniform distribution.
        n_sample : int
            The desired number of negative samples.
        """

        if range_max <= 0:
            raise ValueError("range_max must be a positive integer.")
        if n_sample <= 0:
            raise ValueError("n_sample must be a positive integer.")

        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

            self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels: torch.Tensor):
        """Sample negative samples and calculate their log probabilities.

        Parameters
        ----------
        labels : torch.Tensor, dtype=torch.long, shape=(batch_size,)
            The input labels for which negative samples should be generated.

        Returns
        -------
        true_log_probs : torch.Tensor, dtype=torch.float32, shape=(batch_size,)
            The log probabilities of the input labels according
            to the log-uniform distribution.
        samp_log_probs : torch.Tensor, dtype=torch.float32, shape=(n_samples,)
            The log probabilities of the sampled negative samples according
            to the log-uniform distribution.
        neg_samples : torch.Tensor, dtype=torch.long, shape=(n_samples,)
            The unique negative samples drawn from the log-uniform distribution.

        Raises
        ------
        TypeError
            If `labels` is not a torch.Tensor.
        ValueError
            If `labels` has the wrong dtype, the wrong number of dimensions, is empty,
            or contains values outside the range [0, range_max].
        """

        if not torch.is_tensor(labels):
            raise TypeError("labels must be a torch.Tensor.")
        if labels.dtype != torch.long:
            raise ValueError("labels must be a tensor of dtype long.")
        if labels.dim() > 2 or (labels.dim() == 2 and min(labels.shape) > 1):
            raise ValueError(
                "labels must be a 1-dimensional tensor or a 2-dimensional tensor"
                "with one of the dimensions equal to 1."
            )
        if labels.size(0) == 0:
            raise ValueError("labels must not be an empty tensor.")
        if (labels < 0).any() or (labels > self.range_max).any():
            raise ValueError("All label values must be within the range [0, range_max].")

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)

            return true_log_probs, samp_log_probs, neg_samples


@registry.register("popularity")
class PopularityBasedSampler(nn.Module):
    def __init__(self, max_num_samples: int = 10, seed: Optional[int] = None):
        super().__init__()
        self.labels = torch.ones((1, 1), dtype=torch.int64)
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.negative = self.register_buffer("negative", None)
        self.negative_id = self.register_buffer("negative_id", None)

    def forward(self, positive, positive_id=None):
        if torch.jit.is_scripting():
            raise RuntimeError("PopularityBasedSampler is not supported in TorchScript.")

        if not hasattr(self, "to_call"):
            raise RuntimeError("PopularityBasedSampler must be called after the model")

        del positive, positive_id
        _, _, self.negative_id = self.sampler.sample(self.labels)

        self.negative = self.to_call.embedding_lookup(torch.squeeze(self.negative_id))

        return self.negative, self.negative_id

    def set_to_call(self, to_call):
        self.to_call = to_call
        self.sampler = LogUniformSampler(self.to_call.num_classes, self.max_num_samples)
