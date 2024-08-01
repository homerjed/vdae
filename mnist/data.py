import abc
from typing import Tuple, Callable, Union
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
import torch
from torchvision import transforms, datasets


def expand_if_scalar(x):
    return x[:, jnp.newaxis] if x.ndim == 1 else x


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(self, data, targets, *, key):
        self.data = data 
        self.targets = targets 
        self.key = key

    def loop(self, batch_size):
        dataset_size = self.data.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield self.data[batch_perm], self.targets[batch_perm]
                start = end
                end = start + batch_size


class Scaler:
    forward: Callable 
    reverse: Callable

    def __init__(self, x_min=0., x_max=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


class _TorchDataLoader(_AbstractDataLoader):
    def __init__(self, dataset, num_workers=2, *, key):
        self.dataset = dataset
        self.seed = int(key.sum().item()) 
        self.num_workers = num_workers

    def loop(self, batch_size, num_workers=None):
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers if self.num_workers is not None else num_workers,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        while True:
            for tensor, label in dataloader:
                yield (
                    jnp.asarray(tensor), 
                    expand_if_scalar(jnp.asarray(label))
                )


@dataclass
class ScalerDataset:
    name: str
    train_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    valid_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    scaler: Scaler


def mnist(key: jr.PRNGKey) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    n_pix = 32 # Force power of 2 resolution
    data_shape = (1, n_pix, n_pix)
    context_shape = (1,)

    scaler = Scaler(x_min=0., x_max=1.)

    transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor(), # This magically [0,255] -> [0,1]??
            transforms.Lambda(scaler.forward) # [0,1] -> [-1,1]
        ]
    )
    train_dataset = datasets.MNIST(
        "/project/ls-gruen/users/jed.homer/data/mnist/", 
        train=True,
        download=True, 
        transform=transform
    )
    valid_dataset = datasets.MNIST(
        "/project/ls-gruen/users/jed.homer/data/mnist/", 
        train=False, 
        download=True, 
        transform=transform
    )

    train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
    return ScalerDataset(
        name="mnist",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )