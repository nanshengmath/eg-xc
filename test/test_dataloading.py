from os import path
import warnings
from math import isclose
import pytest
import jax.numpy as jnp
import grain.python as grain

from egxc import dataloading, discretization
from egxc.dataloading import (
    get_dataloaders,
    DatasetEnsemble,
    get_preload_transform,
    PresplitDataset,
    PartiallySplitDataset,
    UnsplitDataset,
    Targets,
)
from egxc.systems.preload import Alignment, PreloadSystem
from egxc.systems import System
from egxc.systems import examples

from egxc.utils.typing import ElectRepTensorType as ERT
from utils import ROOT_DATA_DIR, set_jax_testing_config

from typing import Sequence, Dict

set_jax_testing_config()


@pytest.mark.parametrize(
    'align',
    (Alignment(), Alignment((4, 2, 2), 8, 512)),
    ids=('without padding', 'with padding'),
)
def test_jax_transform(align: Alignment):
    ethanol = examples.get_preloaded('ethanol', 'sto-6g', include_grid=False)
    grid_fn = discretization.get_grid_fn(1, ethanol.atom_z.toset(), align.grid)
    basis_fn = discretization.get_gto_basis_fn('sto-6g', max_period=2, deriv=1)
    transform = dataloading.get_jax_transform((grid_fn, basis_fn), None)
    P0, sys = transform(ethanol)
    assert isinstance(sys, System)
    assert isinstance(P0, jnp.ndarray)


presplit_datasets: Dict[str, dataloading.PresplitDataset] = {}

partially_split_datasets: Dict[str, dataloading.PartiallySplitDataset] = {
    'md17': dataloading.MD17(ROOT_DATA_DIR, 'ethanol'),
    'qm9': dataloading.QM9(ROOT_DATA_DIR, 7),
    '3bpa': dataloading.ThreeBPA(ROOT_DATA_DIR),
}

unsplit_datasets: Dict[str, dataloading.UnsplitDataset] = {
    'des370k': dataloading.DES370K(path.join(ROOT_DATA_DIR, 'des370k')),
}

datasets = presplit_datasets | unsplit_datasets


Dataset = PresplitDataset | PartiallySplitDataset | UnsplitDataset


@pytest.mark.parametrize('dataset', datasets.values(), ids=datasets.keys())
def test__get_item__(dataset: Dataset):
    raw_inputs, targets = dataset[0]
    time = dataset.timed_get_item(1)
    assert time < 0.4, f'It took {time} seconds to get an item from the dataset.'
    assert isinstance(raw_inputs, tuple)
    assert len(raw_inputs) == 4
    assert isinstance(targets, Targets)


@pytest.mark.parametrize(
    'dataset',
    (partially_split_datasets | unsplit_datasets).values(),
    ids=(partially_split_datasets | unsplit_datasets).keys(),
)
def test_dataset_split(dataset: UnsplitDataset):
    if isinstance(dataset, UnsplitDataset):
        train, val, test = dataset.random_split(0.7, 0.1, seed=42)
        assert isclose(len(train), 0.7 * len(dataset), rel_tol=0.001, abs_tol=0.01)
        assert isclose(len(val), 0.1 * len(dataset), rel_tol=0.001, abs_tol=0.01)
        assert isclose(len(test), 0.2 * len(dataset), rel_tol=0.001, abs_tol=0.01)
    else:
        train, val, test = dataset.random_split(0.1, seed=42)

    # test data ensemble
    if isinstance(dataset, UnsplitDataset):
        datasets = DatasetEnsemble.from_random_split(dataset, 0.7, 0.1, 42)
    else:
        datasets = DatasetEnsemble.from_partial_random_split(dataset, 0.1, 42)
    raw_inputs, targets = next(iter(datasets.train))
    assert isinstance(raw_inputs, tuple)
    assert len(raw_inputs) == 4
    assert isinstance(targets, Targets)
    raw_inputs, targets = next(iter(datasets.val))
    assert isinstance(raw_inputs, tuple)
    assert len(raw_inputs) == 4
    assert isinstance(targets, Targets)
    raw_inputs, targets = next(iter(datasets.test))
    assert isinstance(raw_inputs, tuple)
    assert len(raw_inputs) == 4
    assert isinstance(targets, Targets)


def test_md17():
    dataset = partially_split_datasets['md17']
    train, val, test = dataset.random_split(0.1, seed=42)
    assert len(train) == 900
    assert len(val) == 100
    assert len(test) == 1000


def test_qm9():
    dataset = partially_split_datasets['qm9']
    assert len(dataset) == 130831
    sample, targets = dataset[0]
    assert jnp.isclose(targets.energy, -40.47893)  # type: ignore

    train, val, test = dataset.random_split(0.1, seed=42)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(dataset) == len(train) + len(val) + len(test)

    len_dev = len(dataset) - len(test)
    len_val = int(0.1 * len_dev)
    assert len(val) == len_val
    assert len(train) == len_dev - len_val


def test_qm9_exclude_fluorine():
    dataset = dataloading.QM9(ROOT_DATA_DIR, 7, exclude_fluorine=True)
    assert len(dataset) == 128908
    print(len(dataset))

    train, val, test = dataset.random_split(0.1, seed=42)

    len_dev = len(dataset) - len(test)
    len_val = int(0.1 * len_dev)
    assert len(val) == len_val
    assert len(train) == len_dev - len_val


def test_3bpa():
    dataset = partially_split_datasets['3bpa']  # type: ignore
    assert len(dataset) == 13993

    sample, targets = dataset[1000]
    assert jnp.isclose(targets.energy, -17660.06855946407)  # type: ignore

    train, val, test = dataset.random_split(
        0.1,
        seed=42,
        train_subsplits=['train_300K'],  # type: ignore
        test_subsplits=['test_300K', 'test_dih_beta120'],  # type: ignore
    )
    assert len(train) == 450
    assert len(val) == 50
    assert len(test) == 4016


transformations = {
    'restricted+unaligned+exact': get_preload_transform(
        1, 'sto-3g', True, Alignment(), ERT.EXACT, True, False, 1
    ),
    'unrestricted+aligned+df': get_preload_transform(
        1,
        'sto-3g',
        False,
        Alignment((4, 2, 2, 2, 2), 8, 512),
        ERT.DENSITY_FITTED,
        True,
        False,
        1,
    ),
}


@pytest.mark.parametrize('dataset', datasets.values(), ids=datasets.keys())
@pytest.mark.parametrize(
    'transformations', transformations.values(), ids=transformations.keys()
)
def test_datalpoader(dataset: Dataset, transformations: Sequence[grain.Transformation]):
    if isinstance(dataset, PresplitDataset):
        datasets = DatasetEnsemble.from_presplit_dataset(dataset)
    elif isinstance(dataset, PartiallySplitDataset):
        datasets = DatasetEnsemble.from_partial_random_split(dataset, 0.1, 42)
    else:
        assert isinstance(dataset, UnsplitDataset)
        datasets = DatasetEnsemble.from_random_split(dataset, 0.7, 0.1, 42)
    print(next(iter(datasets.train)))
    dataloaders = get_dataloaders(datasets, transformations, True, 0, 42)
    psys, targets = next(iter(dataloaders.train))
    assert isinstance(psys, PreloadSystem)
    assert isinstance(targets, Targets)
    psys, targets = next(iter(dataloaders.val))
    assert isinstance(psys, PreloadSystem)
    assert isinstance(targets, Targets)
    psys, targets = next(iter(dataloaders.test))
    assert isinstance(psys, PreloadSystem)
    assert isinstance(targets, Targets)


def test_no_leakage():
    # TODO: Implement
    warnings.warn('Not implemented')
