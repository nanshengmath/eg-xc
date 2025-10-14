import copy
from typing import Tuple

import numpy as np
import os


from egxc.dataloading import PartiallySplitDataset, RawSample, SupportsIndex, Targets, BaseDataset
from egxc.dataloading.download import download_url, extract_zip

from rdkit import Chem
from tqdm import tqdm

from egxc.dataloading.utils import IndexWrapper

class QM9(PartiallySplitDataset):
    raw_url1 = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    raw_file_names = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    def __init__(
            self,
            data_dir: str,
            heavy_atoms_thresh: int,
            exclude_fluorine = False,
    ):
        data_dir = os.path.join(data_dir, 'qm9')
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.heavy_atoms_thresh = heavy_atoms_thresh
        self.exclude_fluorine = exclude_fluorine

        if exclude_fluorine:
            self.non_fluorine_idxs = np.load(os.path.join(self.processed_dir, 'non_fluorine_idxs.npy'))

        # TODO factor this logic out into BaseDataset?
        complete_file = os.path.join(self.raw_dir, f'complete.marker')
        if not os.path.exists(self.processed_dir) or not os.path.exists(complete_file):
            self.download()
            self.process()
            with open(complete_file, 'w') as f:
                f.write('complete.marker')

    def __getitem__(self, idx: SupportsIndex) -> RawSample:
        if self.exclude_fluorine:
            idx = self.non_fluorine_idxs[idx]
        path = os.path.join(self.processed_dir, 'samples', f'{idx}.npz')  # type: ignore
        data = np.load(path, allow_pickle=True)
        nuc_pos = data['nuc_pos']
        atom_z = data['atom_z']
        targets = Targets(data['energy'], None, None)
        return (nuc_pos, atom_z, 0, 0), targets

    def __len__(self) -> int:
        if self.exclude_fluorine:
            return len(self.non_fluorine_idxs)
        return len(os.listdir(os.path.join(self.processed_dir, 'samples')))

    def download(self) -> None:
        if all([os.path.exists(os.path.join(self.raw_dir, f)) for f in self.raw_file_names]):
            return

        file_path = download_url(self.raw_url1, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        download_url(self.raw_url2, self.raw_dir)
        os.rename(os.path.join(self.raw_dir, '3195404'), os.path.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self) -> None:
        with open(os.path.join(self.raw_dir, self.raw_file_names[2])) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        with open(os.path.join(self.raw_dir, self.raw_file_names[1])) as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = np.asarray(target, dtype=np.float32)
            y = np.concat([y[:, 3:], y[:, :3]], axis=-1)

        atom_z = []
        nuc_pos = []
        energies = []

        suppl = Chem.SDMolSupplier(os.path.join(self.raw_dir, self.raw_file_names[0]), removeHs=False, sanitize=False)

        for i, mol in enumerate(tqdm(suppl)):
            if i not in skip:
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                atom_z.append(atomic_numbers)

                pos = mol.GetConformer().GetPositions()
                pos = np.asarray(pos, dtype=np.float32)
                nuc_pos.append(pos)

                energies.append(y[i, 7])  # u0 column

        out_dir = os.path.join(self.processed_dir, 'samples')
        os.makedirs(out_dir, exist_ok=False)

        heavy_atom_counts = [sum(z > 1 for z in zs) for zs in atom_z]
        non_fluorine_idxs = [i for i, zs in enumerate(atom_z) if not any(z == 9 for z in zs)]

        np.save(os.path.join(self.processed_dir, 'heavy_atom_counts.npy'), np.asarray(heavy_atom_counts))
        np.save(os.path.join(self.processed_dir, 'non_fluorine_idxs.npy'), np.asarray(non_fluorine_idxs))

        for i in range(len(energies)):
            np.savez_compressed(
                os.path.join(out_dir, f'{i}.npz'),
                nuc_pos=nuc_pos[i],
                atom_z=atom_z[i],
                energy=energies[i]
            )

    def random_split(
        self, val_fraction: float, seed: int
    ) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
        heavy_atom_counts = np.load(os.path.join(self.processed_dir, 'heavy_atom_counts.npy'))

        full_idx = range(len(self)) if not self.exclude_fluorine else self.non_fluorine_idxs
        dev_idx = [i for i in full_idx if heavy_atom_counts[i] <= self.heavy_atoms_thresh]
        test_idx = [i for i in full_idx if heavy_atom_counts[i] > self.heavy_atoms_thresh]
        full_set = copy.deepcopy(self)
        dev_set = IndexWrapper(full_set, dev_idx)
        test_set = IndexWrapper(full_set, test_idx)

        n = len(dev_set)
        n_val = int(n * val_fraction)
        n_train = n - n_val
        indices = np.arange(n)
        indices = np.random.RandomState(seed).permutation(indices)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:].tolist()
        train_set = IndexWrapper(dev_set, train_idx)
        val_set = IndexWrapper(dev_set, val_idx)

        return train_set, val_set, test_set
