import numpy as np
import os

from ase.io import read
from egxc.dataloading import PartiallySplitDataset, RawSample, SupportsIndex, Targets, BaseDataset
from egxc.dataloading.download import download_url, extract_zip
from egxc.dataloading.utils import IndexWrapper
from tqdm.auto import tqdm
from typing import List, Tuple


class ThreeBPA(PartiallySplitDataset):
    raw_url = 'https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.1c00647/suppl_file/ct1c00647_si_002.zip'
    train_subsplit_labels = ['train_300K', 'train_mixedT']
    test_subsplit_labels = ['test_300K', 'test_600K', 'test_1200K', 'test_dih_beta120', 'test_dih_beta150', 'test_dih_beta180']
    total_subsplit_labels = train_subsplit_labels + test_subsplit_labels

    def __init__(
            self,
            data_dir: str,
    ):
        data_dir = os.path.join(data_dir, '3bpa')
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')

        complete_file = os.path.join(self.raw_dir, 'complete.marker')
        if not os.path.exists(self.processed_dir) or not os.path.exists(complete_file):
            self.download()
            self.process()
            with open(complete_file, 'w') as f:
                f.write('complete.marker')

        self.subsplit_lens = np.load(os.path.join(self.processed_dir, 'subsplit_lens.npy'))

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        subsplit_idx = 0
        while idx >= self.subsplit_lens[subsplit_idx]:
            idx -= self.subsplit_lens[subsplit_idx]
            subsplit_idx += 1

        path = os.path.join(self.processed_dir, self.total_subsplit_labels[subsplit_idx], f'{idx}.npz')
        data = np.load(path)
        nuc_pos = data['nuc_pos']
        atom_z = data['atom_z']
        targets = Targets(data['energy'], data['nuc_forces'], None)
        return (nuc_pos, atom_z, 0, 0), targets

    def __len__(self) -> int:
        return self.subsplit_lens.sum()

    def download(self) -> None:
        if os.path.exists(os.path.join(self.raw_dir, 'train_300K.xyz')):
            return
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self) -> None:
        subsplit_lens = []
        for subsplit_label in tqdm(self.total_subsplit_labels, desc='processing configurations'):
            out_dir = os.path.join(self.processed_dir, subsplit_label)
            os.makedirs(out_dir, exist_ok=False)
            confs = read(os.path.join(self.raw_dir, subsplit_label + '.xyz'), index=':')
            subsplit_lens.append(len(confs))
            for i, molecule in enumerate(confs):
                np.savez_compressed(
                    os.path.join(out_dir, f'{i}.npz'),
                    nuc_pos=molecule.get_positions(),  # type: ignore
                    atom_z=molecule.get_atomic_numbers(),  # type: ignore
                    energy=molecule.get_potential_energy(),  # type: ignore
                    nuc_forces=molecule.get_forces()  # type: ignore
                )
        np.save(os.path.join(self.processed_dir, 'subsplit_lens.npy'), np.asarray(subsplit_lens))

    def random_split(
        self, val_fraction: float, seed: int, train_subsplits: List[str] = train_subsplit_labels,
            test_subsplits: List[str] = test_subsplit_labels
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        dev_idxs = sum([list(range(self.subsplit_lens[0:i].sum(), self.subsplit_lens[0:(i+1)].sum()))
                                        for i in range(0, 2) if self.train_subsplit_labels[i] in train_subsplits], [])
        test_idxs = sum([list(range(self.subsplit_lens[2:i].sum(), self.subsplit_lens[2:(i+1)].sum()))
                                      for i in range(2, len(self.subsplit_lens)) if self.test_subsplit_labels[i - 2] in test_subsplits], [])
        dev_set = IndexWrapper(self, dev_idxs)
        test_set = IndexWrapper(self, test_idxs)

        n = len(dev_set)
        n_val = int(n * val_fraction)
        n_train = n - n_val
        indices = np.arange(n)
        indices = np.random.RandomState(seed).permutation(indices)
        train_idxs = indices[:n_train].tolist()
        val_idxs = indices[n_train:].tolist()
        train_set = IndexWrapper(dev_set, train_idxs)
        val_set = IndexWrapper(dev_set, val_idxs)

        return train_set, val_set, test_set