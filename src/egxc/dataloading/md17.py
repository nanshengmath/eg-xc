import numpy as onp
import os.path
import copy

from egxc.dataloading import PartiallySplitDataset, RawSample, SupportsIndex, Targets
from egxc.dataloading.base import BaseDataset
from egxc.dataloading.utils import IndexWrapper, random_index_split
from egxc.dataloading.download import download_url, extract_zip
from egxc.utils.constants import KCAL_PER_MOL_TO_HATREE

from typing import Tuple


class MD17(PartiallySplitDataset):
    file_names = {
        'benzene': 'benzene_ccsd_t.zip',
        'aspirin': 'aspirin_ccsd.zip',
        'malonaldehyde': 'malonaldehyde_ccsd_t.zip',
        'ethanol': 'ethanol_ccsd_t.zip',
        'toluene': 'toluene_ccsd_t.zip',
    }
    unique_elements = [1, 6, 8]
    max_period = 2

    def __init__(
        self,
        data_dir: str,
        name: str,
        energy_unit: str = 'hartree',
        distance_unit: str = 'ang',
        train: bool | None = None,
    ):
        assert energy_unit == 'hartree'
        assert distance_unit == 'ang'

        if name not in self.file_names:
            raise ValueError(f"Unknown dataset name '{name}'")
        self.name = name
        data_dir = os.path.join(data_dir, 'md17')
        self.raw_dir = os.path.join(data_dir, name, 'raw')
        self.processed_dir = os.path.join(data_dir, name, 'processed')
        self.train = train

        complete_file = os.path.join(self.raw_dir, 'complete.marker')
        if not os.path.exists(self.processed_dir) or not os.path.exists(complete_file):
            self.download()
            self.process()
            with open(complete_file, 'w') as f:
                f.write('complete.marker')

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        path = os.path.join(self.processed_dir, self.split_str, f'{idx}.npz')  # type: ignore
        data = onp.load(path)
        nuc_pos = data['nuc_pos']
        atom_z = data['atom_z']
        targets = Targets(
            data['energy'] * KCAL_PER_MOL_TO_HATREE,
            data['nuc_forces'] * KCAL_PER_MOL_TO_HATREE,
            None
        )
        return (nuc_pos, atom_z, 0, 0), targets

    def random_split(
        self, val_fraction: float, seed: int
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        train_set = copy.deepcopy(self)
        train_set.train = True
        train_idx, val_idx = random_index_split(
            len(train_set), (1 - val_fraction, val_fraction), seed
        )
        val_set = IndexWrapper(train_set, val_idx)
        train_set = IndexWrapper(train_set, train_idx)
        test_set = copy.deepcopy(self)
        test_set.train = False
        return train_set, val_set, test_set

    @property
    def split_str(self) -> str:
        assert self.train is not None
        return 'train' if self.train else 'test'

    def __len__(self) -> int:
        return len(os.listdir(os.path.join(self.processed_dir, self.split_str)))

    def process(self) -> None:
        for split in ('train', 'test'):
            raw_dir = os.path.join(
                self.raw_dir, self.file_names[self.name].replace('.zip', f'-{split}.npz')
            )
            out_dir = os.path.join(self.processed_dir, split)
            os.makedirs(out_dir, exist_ok=False)
            raw_data = onp.load(raw_dir)

            atom_z = onp.asarray(raw_data['z'])  # CxA
            nuc_pos = onp.asarray(raw_data['R'])  # CxAx3
            energies = onp.asarray(raw_data['E'])  # C
            nuc_forces = onp.asarray(raw_data['F'])  # CxAx3

            for i in range(len(energies)):
                onp.savez_compressed(
                    os.path.join(out_dir, f'{i}.npz'),
                    nuc_pos=nuc_pos[i],
                    atom_z=atom_z,
                    energy=energies[i],
                    nuc_forces=nuc_forces[i],
                )

    def download(self) -> None:
        train_file = self.file_names[self.name].replace('.zip', '-train.npz')
        test_file = self.file_names[self.name].replace('.zip', '-test.npz')
        if os.path.exists(os.path.join(self.raw_dir, train_file)) and \
           os.path.exists(os.path.join(self.raw_dir, test_file)):
            return

        url = f'http://quantum-machine.org/gdml/data/npz/{self.file_names[self.name]}'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
