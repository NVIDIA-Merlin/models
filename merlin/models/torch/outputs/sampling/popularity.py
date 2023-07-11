from typing import Optional

import torch
from torch import nn

from merlin.models.torch.block import registry


class LogUniformSampler(torch.nn.Module):
    """
    LogUniformSampler samples negative samples based on a log-uniform distribution.
    `P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1)`

    This implementation is based on to:
    https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/log_uniform_sampler.py
    TensorFlow Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py

    LogUniformSampler assumes item ids are sorted decreasingly by their frequency.

    if `unique_sampling==True`, then only unique sampled items will be returned.
    The actual # samples will vary from run to run if `unique_sampling==True`,
    as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
    so we use `torch.multinomial(..., replacement=True).unique()` which doesn't guarantee
    the same number of unique sampled items. You can try to increase
    n_samples_multiplier_before_unique to increase the chances to have more
    unique samples in that case.

    Parameters
    ----------
    max_n_samples : int
        The maximum desired number of negative samples. The number of samples might be
        smaller than that if `unique_sampling==True`, as explained above.
    max_id : int
        The maximum value of the range for the log-uniform distribution.
    min_id : Optional[int]
        The minimum value of the range for the log-uniform sampling. By default 0.
    unique_sampling : bool
        Whether to return unique samples. By default True
    n_samples_multiplier_before_unique : int
        If unique_sampling=True, it is not guaranteed that the number of returned
        samples will be equal to max_n_samples, as explained above.
        You can increase n_samples_multiplier_before_unique to maximize
        chances that a larger number of unique samples is returned.
    """

    def __init__(
        self,
        max_n_samples: int,
        max_id: int,
        min_id: Optional[int] = 0,
        unique_sampling: bool = True,
        n_samples_multiplier_before_unique: int = 2,
    ):
        super().__init__()

        if max_id <= 0:
            raise ValueError("max_id must be a positive integer.")
        if max_n_samples <= 0:
            raise ValueError("n_sample must be a positive integer.")

        self.max_id = max_id
        self.unique_sampling = unique_sampling
        self.max_n_samples = max_n_samples
        self.n_sample = max_n_samples
        if self.unique_sampling:
            self.n_sample = int(self.n_sample * n_samples_multiplier_before_unique)

        with torch.no_grad():
            dist = self.get_log_uniform_distr(max_id, min_id)
            self.register_buffer("dist", dist)
            unique_sampling_dist = self.get_unique_sampling_distr(dist, self.n_sample)
            self.register_buffer("unique_sampling_dist", unique_sampling_dist)

    def get_log_uniform_distr(self, max_id: int, min_id: int = 0) -> torch.Tensor:
        """Approximates the items frequency distribution with log-uniform probability distribution
        with P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1).
        It assumes item ids are sorted decreasingly by their frequency.

        Parameters
        ----------
        max_id : int
            Maximum discrete value for sampling (e.g. cardinality of the item id)

        Returns
        -------
        torch.Tensor
            Returns the log uniform probability distribution
        """
        log_indices = torch.arange(1.0, max_id - min_id + 2.0, 1.0).log_()
        probs = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
        if min_id > 0:
            probs = torch.cat(
                [torch.zeros([min_id], dtype=probs.dtype), probs], axis=0
            )  # type: ignore
        return probs

    def get_unique_sampling_distr(self, dist, n_sample):
        """Returns the probability that each item is sampled at least once
        given the specified number of trials. This is meant to be used when
        self.unique_sampling == True.
        That probability can be approximated by by 1 - (1 - p)^n
        and we use a numerically stable version: -expm1(num_tries * log1p(-p))
        """
        return (-(-dist.double().log1p_() * n_sample).expm1_()).float()

    @torch.jit.unused
    def sample(self, labels: torch.Tensor):
        """Sample negative samples and calculate their probabilities.

        If `unique_sampling==True`, then only unique sampled items will be returned.
        The actual # samples will vary from run to run if `unique_sampling==True`,
        as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
        so we use `torch.multinomial(..., replacement=True).unique()`
        which doesn't guarantee the same number of unique sampled items.
        You can try to increase n_samples_multiplier_before_unique
        to increase the chances to have more unique samples in that case.

        Parameters
        ----------
        labels : torch.Tensor, dtype=torch.long, shape=(batch_size,)
            The input labels for which negative samples should be generated.

        Returns
        -------
        neg_samples : torch.Tensor, dtype=torch.long, shape=(n_samples,)
            The unique negative samples drawn from the log-uniform distribution.
        true_probs : torch.Tensor, dtype=torch.float32, shape=(batch_size,)
            The probabilities of the input labels according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        samp_log_probs : torch.Tensor, dtype=torch.float32, shape=(n_samples,)
            The probabilities of the sampled negatives according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        """

        if not torch.is_tensor(labels):
            raise TypeError("Labels must be a torch.Tensor.")
        if labels.dtype != torch.long:
            raise ValueError("Labels must be a tensor of dtype long.")
        if labels.dim() > 2 or (labels.dim() == 2 and min(labels.shape) > 1):
            raise ValueError(
                "Labels must be a 1-dimensional tensor or a 2-dimensional tensor"
                "with one of the dimensions equal to 1."
            )
        if labels.size(0) == 0:
            raise ValueError("Labels must not be an empty tensor.")
        if (labels < 0).any() or (labels > self.max_id).any():
            raise ValueError(
                "All label values must be within the range [0, max_id], ",
                f"got: [{labels.min().item()}, {labels.max().item()}].",
            )

        n_tries = self.n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(
                self.dist, n_tries, replacement=True  # type: ignore
            ).unique()[: self.max_n_samples]

            device = labels.device
            neg_samples = neg_samples.to(device)

            if self.unique_sampling:
                dist = self.unique_sampling_dist
            else:
                dist = self.dist

            true_probs = dist[labels]  # type: ignore
            samples_probs = dist[neg_samples]  # type: ignore

            return neg_samples, true_probs, samples_probs


@registry.register("popularity")
class PopularityBasedSampler(nn.Module):
    """The PopularityBasedSampler generates negative samples for a positive
    input sample based on popularity.

    The class utilizes a LogUniformSampler to draw negative samples from a
    log-uniform distribution. The sampler approximates the items' frequency
    distribution with log-uniform probability distribution. The sampler assumes
    item ids are sorted decreasingly by their frequency.

    Parameters
    ----------
    max_num_samples : int, optional
        The maximum number of negative samples desired. Default is 10.
    unique_sampling : bool, optional
        Whether to return unique samples. Default is True.
    n_samples_multiplier_before_unique : int, optional
        Factor to increase the chances to have more unique samples. Default is 2.

    """

    def __init__(
        self,
        max_num_samples: int = 10,
        unique_sampling: bool = True,
        n_samples_multiplier_before_unique: int = 2,
    ):
        super().__init__()
        self.labels = torch.ones((1, 1), dtype=torch.int64)
        self.max_num_samples = max_num_samples
        self.negative = self.register_buffer("negative", None)
        self.negative_id = self.register_buffer("negative_id", None)
        self.unique_sampling = unique_sampling
        self.n_samples_multiplier_before_unique = n_samples_multiplier_before_unique

    def forward(self, positive, positive_id=None):
        """Computes the forward pass of the PopularityBasedSampler.

        Parameters
        ----------
        positive : torch.Tensor
            The positive samples.
        positive_id : torch.Tensor, optional
            The ids of the positive samples. Default is None.

        Returns
        -------
        negative : torch.Tensor
            The negative samples.
        negative_id : torch.Tensor
            The ids of the negative samples.
        """

        if torch.jit.is_scripting():
            raise RuntimeError("PopularityBasedSampler is not supported in TorchScript.")

        if not hasattr(self, "to_call"):
            raise RuntimeError("PopularityBasedSampler must be called after the model")

        del positive, positive_id
        self.negative_id, _, _ = self.sampler.sample(self.labels)

        self.negative = self.to_call.embedding_lookup(torch.squeeze(self.negative_id))

        return self.negative, self.negative_id

    def set_to_call(self, to_call: nn.Module):
        """Set the model that will utilize this sampler.

        Parameters
        ----------
        to_call : torch.nn.Module
            The model that will utilize this sampler.
        """
        self.to_call = to_call
        self.sampler = LogUniformSampler(
            max_n_samples=self.max_num_samples,
            max_id=self.to_call.num_classes,
            unique_sampling=self.unique_sampling,
            n_samples_multiplier_before_unique=self.n_samples_multiplier_before_unique,
        )
