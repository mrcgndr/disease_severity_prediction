from itertools import product
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


def safe_multinomial(input: torch.Tensor, num_samples: int, replacement: bool) -> torch.Tensor:
    """Performs a multinomial sampling. The function chooses between using PyTorch's `torch.multinomial`
    or NumPy's `np.random.choice` based on the length of the input tensor.

    Args:
        input (torch.Tensor): The probability distribution over classes.
        num_samples (int): The number of samples to draw from the distribution.
        replacement (bool): Whether to draw samples with replacement.

    Returns:
        torch.Tensor: A tensor containing the indices of the drawn samples.
    """
    if len(input) <= 2**24:
        return torch.multinomial(input=input, num_samples=num_samples, replacement=replacement)
    else:
        input = np.asarray(input / input.sum())
        return torch.as_tensor(np.random.choice(a=len(input), size=num_samples, replace=replacement, p=input))


def calculate_weights(targets: Iterable[torch.Tensor]) -> torch.Tensor:
    """Calculates the weights of data points by its label abundance. For multiple
    labels, the weights are combined by multiplication.

    Args:
        targets (List[torch.Tensor]): List of label tensors.

    Returns:
        torch.Tensor: Tensor of weights for each data point.
    """
    weights = [dict((t.item(), torch.tensor([1 / (target == t).sum()])) for t in torch.unique(target)) for target in targets]
    weights_sum = [np.sum([t.item() for t in w.values()]) for w in weights]
    samples_weight = torch.ones(len(targets[0]))
    for target, weight, sum in zip(targets, weights, weights_sum):
        samples_weight *= torch.as_tensor([weight[t.item()] for t in target]) / sum
    samples_weight /= samples_weight.max()
    return samples_weight


def balanced_train_val_split(
    dataset: Dataset, val_size: float, balance_by: Optional[Iterable[str]] = None, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a dataset balanced by a feature

    Args:
        dataset (Dataset): Dataset to be splitted.
        val_size (float): Validation dataset size.
        balance_by (Optional[str], optional): Feature that sould be balanced by. Defaults to None.
        seed (Optional[int], optional): If given, random seed is fixed. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensor with training and validation data indices.
    """
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    if balance_by:
        targets = [dataset.get_all(b) for b in balance_by]
        unique_targets = [np.unique(t) for t in targets]
        idx = np.arange(len(targets[0]))

        train_indices = []
        val_indices = []
        for t in product(*unique_targets):
            masks = np.asarray([target == t_ for target, t_ in zip(targets, t)])
            pick = masks.sum(axis=0) == len(targets)
            n_picks = pick.sum()
            if n_picks > 0:
                rand_idx = np.random.choice(idx[pick], n_picks, replace=False)
                split = int(np.floor(val_size * n_picks))
                train_indices.extend(rand_idx[split:])
                val_indices.extend(rand_idx[:split])
        train_indices = torch.as_tensor(train_indices)[torch.randperm(len(train_indices))]
        val_indices = torch.as_tensor(val_indices)[torch.randperm(len(val_indices))]
    else:
        # randomized = torch.multinomial(input=torch.ones(len(dataset)), num_samples=len(dataset), replacement=False)
        randomized = torch.randperm(len(dataset))
        split = int(np.floor(val_size * len(dataset)))
        train_indices, val_indices = randomized[split:], randomized[:split]

    return train_indices, val_indices


class ImbalancedDatasetSampler(Sampler):
    """Sampler for imbalanced datasets."""

    def __init__(self, targets: Iterable[torch.Tensor], num_samples: int = None):
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(targets[0]) if num_samples is None else num_samples

        self.targets = targets
        self.weights = calculate_weights(self.targets)

    def __iter__(self) -> iter:
        rand = safe_multinomial(self.weights, self.num_samples, replacement=True)
        yield from iter(rand.tolist())

    def __len__(self) -> int:
        return self.num_samples


class DistributedImbalancedDatasetSampler(Sampler):
    """Sampler for imbalanced dataset with support for distributed computing."""

    def __init__(self, targets: Iterable[torch.Tensor], num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.targets = targets
        self.weights = calculate_weights(self.targets)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.targets[0]) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.targets[0]), generator=g).tolist()
        else:
            indices = list(range(len(self.targets[0])))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        rand = safe_multinomial(self.weights[indices], self.num_samples, replacement=True)

        yield from iter(torch.tensor(indices)[rand].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
