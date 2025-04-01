import pandas as pd
import numpy as onp
from openqdc import datasets as oqdc_datasets

from egxc.dataloading.base import (
    BaseDataset,
    UnsplitDataset,
    Targets,
    RawSample,
    SupportsIndex,
)

from egxc.dataloading.utils import IndexWrapper, random_index_split
from typing import Tuple


class OQDCdataset(BaseDataset):
    data: pd.DataFrame

    def __init__(
        self,
        data_dir: str,
        energy_unit: str = 'hartree',
        distance_unit: str = 'ang',
    ):
        self.energy_unit = energy_unit
        self.distance_unit = distance_unit
        self.init_dataframe(data_dir)
        self.unique_elements = self.data['atomic_numbers'].explode().unique()  # type: ignore
        max_z = max(self.unique_elements)
        rows = onp.array((2, 10, 18, 36, 54, 86, 118))
        self.max_period = int((max_z > rows).sum() + 1)

    def init_dataframe(self, data_dir: str):
        raise NotImplementedError


class DES370K(OQDCdataset, UnsplitDataset):
    def init_dataframe(self, data_dir: str):
        self.data_dir = data_dir
        data = oqdc_datasets.DES370K(
            energy_unit=self.energy_unit,
            distance_unit=self.distance_unit,
            cache_dir=data_dir,
        )
        self.data = pd.DataFrame(data)

    def partition_data(self, idx_sequence: pd.Series) -> pd.DataFrame:  # type: ignore
        return self.data.iloc[idx_sequence]

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        row = self.data.iloc[idx]  # type: ignore
        nuc_pos = row['positions']
        atom_z = row['atomic_numbers']
        charge = sum(
            set(row['charges'])
        )  # TODO: check if this is correct (wierd charge format)
        number_of_electrons = sum(row['atomic_numbers']) - charge
        spin = number_of_electrons % 2
        total_energy = None  # TODO: calculate total energies?
        targets = Targets(total_energy, None, None)
        return (nuc_pos, atom_z, charge, spin), targets

    def random_split(
        self, train_fraction: float, val_fraction: float, seed: int
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        train_idx, val_idx, test_idx = random_index_split(
            len(self),
            (train_fraction, val_fraction, 1 - train_fraction - val_fraction),
            seed,
        )
        train = IndexWrapper(self, train_idx)
        val = IndexWrapper(self, val_idx)
        test = IndexWrapper(self, test_idx)
        return train, val, test
