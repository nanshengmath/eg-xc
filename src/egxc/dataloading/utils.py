import numpy as onp
from grain.python import RandomAccessDataSource
from grain._src.python.data_sources import SupportsIndex

from egxc.dataloading.base import RawSample, BaseDataset

from typing import List, Sequence, Tuple


def random_index_split(
    num_elements: int, fractions: Sequence[float], seed: int
) -> Tuple[List[int], ...]:
    sum_fractions = sum(fractions)
    assert sum_fractions == 1.0
    indices = onp.arange(num_elements)
    indices = onp.random.RandomState(seed).permutation(indices)
    out = []
    previous = 0
    for frac in fractions[:-1]:
        n = int(num_elements * frac)
        out.append(indices[previous : previous + n].tolist())
        previous += n
    out.append(indices[previous:].tolist())
    return tuple(out)



class IndexWrapper(BaseDataset):
    def __init__(self, dataset: RandomAccessDataSource, indices: List[int]):
        self.__dataset = dataset
        self.__indices = indices

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        idx = self.__indices[idx]
        return self.__dataset[idx]

    def __len__(self) -> int:
        return len(self.__indices)