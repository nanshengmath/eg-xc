from time import time
from dataclasses import dataclass
from grain.python import RandomAccessDataSource
from grain._src.python.data_sources import SupportsIndex

from egxc.utils.typing import NpIntA, NpFloatAx3, NpFloatBxB

from typing import Tuple, NamedTuple, List, Any


Data = Any
Charge = int
Spin = int


class Targets(NamedTuple):
    energy: float | None
    nuc_forces: NpFloatAx3 | None
    density_matrix: NpFloatBxB | None


RawInput = Tuple[NpFloatAx3, NpIntA, Charge, Spin]
RawSample = Tuple[RawInput, Targets]


class BaseDataset(RandomAccessDataSource):
    # TODO: consider caching the data on the worker node using tempfile to avoid networking overhead
    data: Data
    max_period: int
    unique_elements: List[int]

    energy_unit: str
    distance_unit: str

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        """
        This abstract method needs to be implemented by the child classes
        as if it were a __getitem__ method of a singular dataset.
        Which might be counterintuitive, for the split datasets.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def timed_get_item(self, idx: int) -> float:
        time_start = time()
        self[idx]
        time_end = time()
        return time_end - time_start


class PresplitDataset(BaseDataset):
    @property
    def split(self) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        """
        Abstract method to return the train, val, and test datasets
        """
        raise NotImplementedError


class PartiallySplitDataset(BaseDataset):
    """
    Datasets with preexisting splits for train and test sets
    but without a validation set.
    """

    def random_split(
        self, val_fraction: float, seed: int
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        """
        Abstract method to split the dataset into train, val, and test sets
        """
        raise NotImplementedError


class UnsplitDataset(BaseDataset):
    def random_split(
        self, train_fraction: float, val_fraction: float, seed: int
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        """
        Abstract method to split the dataset into train, val, and test sets
        """
        raise NotImplementedError


@dataclass
class DatasetEnsemble:
    train: BaseDataset
    val: BaseDataset
    test: BaseDataset

    @classmethod
    def infer_split(
        cls,
        dataset: BaseDataset,
        train_fraction: float | None = None,
        val_fraction: float | None = None,
        seed: int = 0,
    ) -> 'DatasetEnsemble':
        if isinstance(dataset, PresplitDataset):
            assert train_fraction is None and val_fraction is None
            return cls(*dataset.split)
        if isinstance(dataset, PartiallySplitDataset):
            assert train_fraction is None and val_fraction is not None
            return cls(*dataset.random_split(val_fraction, seed))
        if isinstance(dataset, UnsplitDataset):
            assert train_fraction is not None and val_fraction is not None
            return cls(*dataset.random_split(train_fraction, val_fraction, seed))
        raise ValueError(f'Unknown dataset type: {type(dataset)}')

    @classmethod
    def from_presplit_dataset(cls, dataset: PresplitDataset) -> 'DatasetEnsemble':
        return cls(*dataset.split)

    @classmethod
    def from_partial_random_split(
        cls,
        dataset: PartiallySplitDataset,
        val_fraction: float,
        seed: int,
    ) -> 'DatasetEnsemble':
        train, val, test = dataset.random_split(val_fraction, seed)
        return cls(train, val, test)

    @classmethod
    def from_random_split(
        cls,
        dataset: UnsplitDataset,
        train_fraction: float,
        val_fraction: float,
        seed: int,
    ) -> 'DatasetEnsemble':
        train, val, test = dataset.random_split(train_fraction, val_fraction, seed)
        return cls(train, val, test)
