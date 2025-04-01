from dataclasses import dataclass
import grain.python as grain

from egxc.systems import PreloadSystem
from egxc.dataloading.base import DatasetEnsemble, BaseDataset

from typing import Sequence


class GrainDataLoaderWrapper:
    """
    Wrapper around a grain DataLoader to provide a length property, such that the iteration
    stops after the specified number of samples, but the worker processes continue to run
    until the dataloader has been run over the specified number of total epochs.
    """

    __iterator = None
    __counter = 0

    def __init__(self, dataloader: grain.DataLoader, length: int):
        self.__dataloader = dataloader
        self.__length = length

    def __iter__(self):
        if self.__iterator is None:
            # only initialize the iterator once
            self.__iterator = iter(self.__dataloader)
        return self

    def __len__(self):
        return self.__length

    def __next__(self):
        self.__counter += 1
        if self.__counter > self.__length:
            self.__counter = 0
            raise StopIteration
        return next(self.__iterator)  # type: ignore


@dataclass
class DataLoaders:
    train: GrainDataLoaderWrapper
    val: GrainDataLoaderWrapper
    test: GrainDataLoaderWrapper


def get_sample_for_model_init(
    dataset: BaseDataset, preload_transform: Sequence[grain.Transformation]
) -> PreloadSystem:
    """
    Utility function to get a single sample from a dataset for model initialization.
    Uses the exact same transformations as the main dataloaders, but avoids thread / process
    creation overhead.
    """
    sampler = grain.SequentialSampler(1, shard_options=grain.NoSharding())
    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=preload_transform,
        sampler=sampler,
        worker_count=0,
    )
    return next(iter(dataloader))[0]


def get_dataloaders(
    datasets: DatasetEnsemble,
    transformations: Sequence[grain.Transformation],
    shuffle: bool,
    workers: int | None,
    random_seed: int,
) -> DataLoaders:
    """
    Get dataloaders for training, validation, and testing.

    Args:
        datasets: Tuple of datasets for training, validation, and testing.
    """

    def get_dataloader(dataset: BaseDataset) -> GrainDataLoaderWrapper:
        length = len(dataset)
        sampler = grain.IndexSampler(
            num_records=length,
            shard_options=grain.NoSharding(),
            shuffle=shuffle,
            seed=random_seed,
        )

        dataloader = grain.DataLoader(
            data_source=dataset,
            operations=transformations,
            sampler=sampler,
            worker_count=workers,
        )
        return GrainDataLoaderWrapper(dataloader, length)

    out = DataLoaders(
        get_dataloader(datasets.train),
        get_dataloader(datasets.val),
        get_dataloader(datasets.test),
    )
    return out
