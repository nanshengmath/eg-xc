"""
This module contains functions to build sample molecules for testing purposes.
"""

import numpy as np

from .preload import PreloadSystem, preload_system_using_pyscf
from .base import System
from egxc.utils.typing import ElectRepTensorType as ERT
from egxc.utils.typing import Alignment

from numpy.typing import ArrayLike


def __create_sys(
    nuc_pos: ArrayLike,
    atom_z: ArrayLike,
    alignment: int | Alignment,
    basis: str,
    charge: int = 0,
    spin: int | None = None,
    center: bool = False,
    include_fock_tensors: bool = True,
    ert_type: ERT = ERT.DENSITY_FITTED,
    include_grid: bool = True,
    spin_restricted: bool | None = None,
) -> PreloadSystem:
    if spin is None:
        n_electrons = np.sum(atom_z) - charge
        spin = n_electrons % 2

    if spin_restricted is None:
        spin_restricted = spin == 0
    elif spin_restricted and spin != 0:
        raise ValueError('Spin restricted system must have spin=0')

    if type(alignment) is Alignment:
        alignment = alignment
    elif alignment > 1:  # type: ignore
        alignment = Alignment(alignment, alignment, 128 * alignment) # type: ignore
    else:
        alignment = Alignment()

    psys = preload_system_using_pyscf(
        nuc_pos=nuc_pos,
        atom_z=atom_z,
        charge=charge,
        spin=spin,  # type: ignore
        basis=basis,
        spin_restricted=spin_restricted,
        alignment=alignment,
        center=center,
        include_fock_tensors=include_fock_tensors,
        ert_type=ert_type,
        include_grid=include_grid,
    )
    return psys


def get_preloaded(
    name: str | int,
    basis: str,
    alignment: int | Alignment = 4,
    include_grid: bool = True,
    ert_type: ERT = ERT.DENSITY_FITTED,
    spin_restricted: bool | None = None,
) -> PreloadSystem:
    """
    Builds a few sample molecules for testing purposes.
    """

    sys_kwargs = {
        'alignment': alignment,
        'basis': basis,
        'include_grid': include_grid,
        'ert_type': ert_type,
        'spin_restricted': spin_restricted,
    }

    if type(name) is int:
        sys_kwargs['atom_z'] = np.arange(1, name)
        sys_kwargs['nuc_pos'] = np.stack(
            [np.zeros(name), np.zeros(name), np.linspace(0, 2 * name, name)], axis=-1
        )
    elif type(name) is str:
        name = name.lower()

        if name == 'h':
            sys_kwargs['atom_z'] = np.array([1])
            sys_kwargs['nuc_pos'] = np.array([[0.0, 0.0, 0.0]])
            sys_kwargs['spin'] = 1
        elif name == 'h2+':
            sys_kwargs['atom_z'] = np.array([1, 1])
            sys_kwargs['nuc_pos'] = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
            sys_kwargs['charge'] = 1
            sys_kwargs['spin'] = 1
        elif name is None or name == 'h2':
            sys_kwargs['atom_z'] = np.array([1, 1])
            sys_kwargs['nuc_pos'] = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        elif name == 'o2':
            sys_kwargs['atom_z'] = np.array([8, 8])
            sys_kwargs['nuc_pos'] = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        elif name == 'organic':
            sys_kwargs['atom_z'] = np.array([1, 6, 7, 8, 9])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                ]
            )
        elif name == 'water' or name == 'h2o':
            r"""Single water molecule
            Structure of single water molecule calculated with DFT using B3LYP
            functional and 6-31+G** basis set <https://cccbdb.nist.gov/>"""
            sys_kwargs['atom_z'] = np.array([8, 1, 1])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [0.0000, 0.0000, 0.1165],
                    [0.0000, 0.7694, -0.4661],
                    [0.0000, -0.7694, -0.4661],
                ]
            )
        elif name == 'ethanol':
            # 0-th conformation from md17
            sys_kwargs['atom_z'] = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [-0.14365933, -0.11813374, -0.56843375],
                    [-0.17613, 1.28513285, -0.00859315],
                    [0.21953989, -1.07718569, 0.51279995],
                    [-1.17635944, -0.44982293, -1.02836848],
                    [0.49325929, -0.34332167, -1.39655428],
                    [-0.82220941, 1.53936818, 0.84187767],
                    [0.86485323, 1.37019756, 0.45488715],
                    [-0.22079702, 2.03512503, -0.80257909],
                    [1.18606471, -0.96054634, 0.66677837],
                ]
            )
        elif name == 'aspirin':
            # 0-th conformation from md17
            sys_kwargs['atom_z'] = np.array(
                [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            )
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [2.15275078, -0.93790121, -0.05378575],
                    [0.99956719, 1.13262738, -1.67300307],
                    [2.73218273, -0.44859684, -1.19275553],
                    [2.14794307, 0.41880283, -2.08405233],
                    [-3.15705489, 1.42240939, 0.33067654],
                    [0.91168856, -0.33727827, 0.29772754],
                    [0.36105629, 0.72618343, -0.42339745],
                    [-0.40166094, -0.12259909, 2.26219435],
                    [-2.13128849, -0.48888369, -0.80224462],
                    [0.29867456, -2.24990948, 1.44246368],
                    [0.1239993, -0.83296539, 1.42230211],
                    [-2.04596577, 0.64616435, -0.22107209],
                    [-0.88096468, 1.36265193, -0.06099633],
                    [-0.02472582, -2.46371902, 2.33402192],
                    [2.47956412, -1.70739289, 0.55101985],
                    [0.49126967, 1.99943374, -2.09234064],
                    [3.77819263, -0.85891833, -1.40897491],
                    [2.75656656, 0.58901616, -2.98423316],
                    [-2.82095412, 2.33964098, 0.93973904],
                    [-3.76033286, 1.76801371, -0.50670433],
                    [-3.80678333, 0.80577181, 0.93093152],
                ]
            )
        elif name == 'benzene':
            # 0-th conformation from md17
            sys_kwargs['atom_z'] = np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [-1.17701667, 0.628175, -0.3431],
                    [0.03758333, 1.274275, -0.5205],
                    [1.21898333, 0.650375, -0.2513],
                    [1.24578333, -0.654725, 0.3335],
                    [-0.01361667, -1.257825, 0.5988],
                    [-1.22571667, -0.640425, 0.2372],
                    [-2.11961667, 1.163575, -0.5769],
                    [0.05898333, 2.257775, -1.0766],
                    [2.16458333, 1.159175, -0.3528],
                    [2.16518333, -1.198125, 0.4236],
                    [-0.10771667, -2.264025, 1.086],
                    [-2.24741667, -1.118225, 0.4421],
                ]
            )
        elif name == 'toluene':
            # 0-th conformation from md17
            sys_kwargs['atom_z'] = np.array([6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [2.21099309, -0.213506861, 0.249508229],
                    [0.681932229, 0.000127978667, -0.126283431],
                    [-0.0239355113, -1.13700971, -0.553183121],
                    [-1.39124473, -0.959111431, -0.512795811],
                    [-2.12171677, 0.210078419, -0.0963132607],
                    [-1.37364578, 1.34153136, 0.195166499],
                    [0.0826932687, 1.19693882, 0.204563959],
                    [2.62700104, -1.00310271, -0.191848021],
                    [2.87154393, 0.560016199, 0.0649209093],
                    [2.19943319, -0.566922311, 1.22136951],
                    [0.564269529, -2.14756659, -0.690916221],
                    [-2.03206059, -1.82119392, -0.737063061],
                    [-3.23640748, 0.181348219, -0.141217201],
                    [-1.75321084, 2.40497179, 0.417926179],
                    [0.694355439, 1.95340076, 0.696164839],
                ]
            )
        elif name == 'malonaldehyde':
            # 0-th conformation from md17
            sys_kwargs['atom_z'] = np.array([6, 6, 6, 8, 8, 1, 1, 1, 1])
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [0.92768327, -0.76532832, 0.43548246],
                    [-0.55217159, -0.17860673, 0.3188344],
                    [-0.38930302, 0.82837453, -0.7735311],
                    [-1.14209009, 0.92354304, -1.78297229],
                    [1.82359972, 0.06762339, 0.36430571],
                    [1.05790473, -1.94286082, 0.55973303],
                    [-1.38445232, -0.97464033, 0.27831208],
                    [-0.75663989, 0.3534461, 1.32381917],
                    [0.41546923, 1.68844912, -0.7239835],
                ]
            )
        elif name == '3bpa':
            # fmt: off
            sys_kwargs['atom_z'] = np.array([6, 6, 6, 1, 6, 8, 7, 7, 1, 1, 6, 1, 1, 6, 6, 1, 1, 6, 6, 6, 1, 6, 1, 6, 1, 1, 1])
            # fmt: on
            sys_kwargs['nuc_pos'] = np.array(
                [
                    [1.38963128, 1.30881270, -1.84807340],
                    [0.25134028, 1.61695797, -1.17378268],
                    [2.34174165, 2.30279245, -2.05237245],
                    [1.74670811, 0.30770268, -2.08813647],
                    [0.05520299, 2.95681402, -0.72831891],
                    [-0.81096017, 0.83736826, -0.91802960],
                    [-1.10099896, 3.26937222, -0.10832467],
                    [0.91391981, 3.92808926, -0.99610811],
                    [-1.73558122, 2.55577479, 0.18972739],
                    [-1.19481836, 4.21421525, 0.12159553],
                    [2.07766676, 3.62803433, -1.63042725],
                    [2.81650095, 4.38833396, -1.67628005],
                    [3.23433192, 2.10381485, -2.63100063],
                    [-0.57931678, -0.63039174, -0.91326348],
                    [-1.77506852, -1.28327640, -0.40022312],
                    [0.20195626, -0.97701420, -0.35217348],
                    [-0.31077873, -0.89430482, -2.04282024],
                    [-3.00526392, -1.38331587, -1.13839590],
                    [-1.88488065, -1.72681611, 0.94414745],
                    [-4.15739837, -2.00672026, -0.71479091],
                    [-2.95735117, -1.01520162, -2.24735924],
                    [-4.19469156, -2.42753431, 0.61887066],
                    [-5.04818305, -2.24955496, -1.33679401],
                    [-3.08464301, -2.27809040, 1.38882121],
                    [-5.07725553, -2.96292210, 1.05906295],
                    [-3.24228124, -2.56751763, 2.40845989],
                    [-1.09224915, -1.37255578, 1.68160820],
                ]
            )
        else:
            raise NotImplementedError(f'No structure registered for: {name}')

    return __create_sys(**sys_kwargs)


def get(
    name: str | int,
    basis: str,
    alignment: int | Alignment = 4,
    include_grid: bool = True,
    ert_type: ERT = ERT.DENSITY_FITTED,
    spin_restricted: bool | None = None,
) -> System:
    """
    Builds a few sample molecules for testing purposes.
    """
    return System.from_preloaded(
        get_preloaded(name, basis, alignment, include_grid, ert_type, spin_restricted)
    )


def cubic_hydrogen(n: int, align: bool = False, basis: str = 'sto-3g') -> PreloadSystem:
    """
    Builds a Structure of hydrogen atoms arranged in a simple cubic lattice.

    Args:
        n (int): The number of hydrogen atoms for the cubic cell. For example, n=4 will
        build a 4x4x4 cubic lattice.

    Raises:
        ValueError: If n is less than 1.

    Returns:
        Structure: A Structure object representing the cubic lattice of hydrogen atoms.
    """
    if n < 1:
        raise ValueError('Expect at least one hydrogen atom in cubic lattice')

    b = 1.4 * np.arange(0, n)
    pos = np.stack(np.meshgrid(b, b, b)).reshape(3, -1).T
    pos = np.round(pos - np.mean(pos, axis=0), decimals=3)
    return __create_sys(
        pos, np.ones(pos.shape[0], dtype=np.int64), alignment=align, basis=basis
    )


# TODO: implement
# @requires_package("pyquante2")
# def from_pyquante(name: str) -> Structure:
#     """Load molecular structure from pyquante2.geo.samples module

#     Args:
#         name (str): Possible names include ch4, c6h6, aspirin, caffeine, hmx, petn,
#                     prozan, rdx, taxol, tylenol, viagara, zoloft

#     Returns:
#         Structure
#     """
#     from pyquante2.geo import samples

#     pqmol = getattr(samples, name)
#     atomic_number, position = zip(*[(a.Z, a.r) for a in pqmol])
#     atomic_number, position = [np.asarray(x) for x in (atomic_number, position)]
#     return create(atomic_number, position)
