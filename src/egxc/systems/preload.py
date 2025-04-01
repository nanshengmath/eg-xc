from dataclasses import dataclass
from pyscf import gto, df, dft
from pyscf.dft import gen_grid
import numpy as onp
import einops
from scipy import linalg

from egxc.utils.constants import ANGSTROM_TO_BOHR
from egxc.utils.typing import (
    Alignment,
    PRECISION,
    ElectRepTensorType,
    CompileStaticIntA,
    CompileStaticStr,
    CompileStaticInt,
    PermutiationInvariantHashableArray as PerInvHashArray,
)
from egxc.utils import pad

from numpy.typing import ArrayLike

Array = onp.ndarray


def z_to_periods(Z: Array) -> CompileStaticIntA:
    ROWS = onp.array((2, 10, 18, 36, 54, 86, 118))
    rows = einops.repeat(ROWS, 'r -> n r', n=len(Z))
    Z = einops.repeat(Z, 'n -> n r', r=len(ROWS))
    p = (Z > rows).sum(axis=1) + 1
    return tuple(p.tolist())


def compute_electron_occupancy(
    spin: int, n_electrons: int, n_basis_functions: int, spin_restricted: bool
) -> Array:
    B = n_basis_functions
    if spin_restricted:
        assert spin % 2 == 0, 'Spin must be even for restricted'
        occ = onp.zeros(B, dtype=onp.uint8)
        occ[: n_electrons // 2] = 2
    else:
        n_up = n_electrons // 2
        n_down = n_electrons - n_up
        occ_up = onp.zeros(B, dtype=onp.uint8)
        occ_up[:n_up] = 1
        occ_down = onp.zeros(B, dtype=onp.uint8)
        occ_down[:n_down] = 1
        occ = onp.stack((occ_up, occ_down), dtype=onp.uint8)
    return occ


def compute_electron_repulsion_tensor(
    mol: gto.Mole,
    ert_type: ElectRepTensorType,
    mask: Array | None,
    aux_mask: Array | None = None,
    aux_basis: str = 'weigend',  # TODO: check other aux basis
):
    if ert_type == ElectRepTensorType.EXACT:
        ert = mol.intor('int2e')
        if mask is not None:
            ert = onp.einsum('ijkl,i,j,k,l-> ijkl', ert, mask, mask, mask, mask)
    elif ert_type == ElectRepTensorType.DENSITY_FITTED:
        auxmol = df.addons.make_auxmol(mol, aux_basis)
        nao = mol.nao
        naux = auxmol.nao
        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(
            nao * nao, naux
        )
        ints_2c2e = auxmol.intor('int2c2e')
        cho = linalg.cholesky(ints_2c2e)
        ert = linalg.solve_triangular(cho.T, ints_3c2e.T, lower=True)
        ert = ert.reshape(naux, nao, nao)
        if mask is not None:
            assert (
                aux_mask is not None
            ), 'aux_mask must be provided'  # TODO: do we actually need this
            ert = onp.einsum('qij,q,i,j-> qij', ert, aux_mask, mask, mask)
    return ert


@dataclass(frozen=True)
class PreloadFockTensors:
    """
    Tensors that are constant for a given Structure and basis set which are
    used to calculate the Fock matrix.
    """

    basis_mask: Array
    overlap: Array
    core_hamiltonian: Array
    electron_repulsion_tensor: Array
    occupancies: Array

    def __repr__(self) -> str:
        return (
            f'##### PreloadFockTensors ##### \n'
            f'# basis mask =\n{self.basis_mask},\n'
            f'# Overlap Matrix S =\n{self.overlap},\n'
            f'# Core Hamiltonian H = \n{self.core_hamiltonian},\n'
            f'# Orbital Occupations =\n{self.occupancies},\n'
            f'# Electron Repulsion Tensor (shape: {self.electron_repulsion_tensor.shape})'
        )


def preload_fock_tensors_using_pyscf(
    mol: gto.Mole,
    spin: int,
    n_electrons: int,
    spin_restricted: bool,
    ert_type: ElectRepTensorType,
    alignment: Alignment,
) -> PreloadFockTensors:
    B = mol.nao  # number of basis functions
    overlap = mol.intor('int1e_ovlp')
    core_hamiltonian = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    ert = compute_electron_repulsion_tensor(mol, ert_type, None)
    occupancies = compute_electron_occupancy(spin, n_electrons, B, spin_restricted)
    basis_mask = onp.ones(B, dtype=bool)
    if alignment.is_aligned:
        assert onp.all(basis_mask), 'Fock tensors should only be padded once'
        b_pad = pad.calc_padding_size(B, alignment.basis)
        basis_mask = onp.pad(basis_mask, (0, b_pad))

        overlap = onp.pad(overlap, ((0, b_pad), (0, b_pad)))
        overlap[~basis_mask, ~basis_mask] = 1.0

        core_hamiltonian = onp.pad(core_hamiltonian, ((0, b_pad), (0, b_pad)))
        # pad diagonal with large non-equal values to avoid issues in the generalized eigenvalue problem of the Fock matrix
        core_hamiltonian[~basis_mask, ~basis_mask] = onp.arange(10_000, 10_000 + 1000 * b_pad, 1000)

        if ert.ndim == 4:
            ert = onp.pad(ert, ((0, b_pad), (0, b_pad), (0, b_pad), (0, b_pad)))
        else:  # pad density-fitted electron repulsion tensor
            Q = ert.shape[0]
            aux_b_pad = pad.calc_padding_size(Q, alignment.basis)
            ert = onp.pad(ert, ((0, aux_b_pad), (0, b_pad), (0, b_pad)))

        if occupancies.ndim == 1:  # spin-restricted
            occupancies = onp.pad(occupancies, (0, b_pad))
        else:  # spin-unrestricted
            occupancies = onp.pad(occupancies, ((0, 0), (0, b_pad)))

    return PreloadFockTensors(basis_mask, overlap, core_hamiltonian, ert, occupancies)


@dataclass(frozen=True)
class PreloadGrid:
    coords: Array  # FloatNx3
    weights: Array  # FloatN
    aos: Array  # FloatNxB
    grad_aos: Array | None = None  # FloatNxBx3

    @classmethod
    def create(
        cls,
        coords: Array,
        weights: Array,
        aos: Array,
        grad_aos: Array | None,
        alignment: Alignment,
    ) -> 'PreloadGrid':
        if alignment.is_aligned:
            assert isinstance(alignment.atom, int), 'Atom alignment must be an integer'
            N, B = aos.shape
            n_pad = pad.calc_padding_size(N, alignment.grid)
            b_pad = pad.calc_padding_size(B, alignment.basis)
            coords = onp.pad(coords, ((0, n_pad), (0, 0)))  # type: ignore
            weights = onp.pad(weights, (0, n_pad))  # type: ignore
            aos = onp.pad(aos, ((0, n_pad), (0, b_pad)), mode='edge')
            if grad_aos is not None:
                grad_aos = onp.pad(
                    grad_aos, ((0, n_pad), (0, b_pad), (0, 0)), mode='edge'
                )
        return cls(coords, weights, aos, grad_aos)


def preload_grid_using_pyscf(
    mol: gto.Mole, grids: gen_grid.Grids, grid_level: int, alignment: Alignment
) -> PreloadGrid:
    grids.level = grid_level
    grids.build()
    coords = grids.coords
    weights = grids.weights
    aos_and_grad_aos = mol.eval_gto('GTOval_sph_deriv1', coords)
    aos = aos_and_grad_aos[0]
    grad_aos = einops.rearrange(aos_and_grad_aos[1:], 's n b -> n b s')
    return PreloadGrid.create(coords, weights, aos, grad_aos, alignment)  # type: ignore


@dataclass(frozen=True)
class PreloadSystem:
    """
    Frozen jit compatible dataclass used for cpu-based preloading operations
    for the subsequent construction of gpu-based System objects.
    """

    nuc_pos: Array  # nuclei positions
    atom_z: PerInvHashArray  # atomic numbers
    atom_mask: Array
    fock_tensors: PreloadFockTensors | None
    grid: PreloadGrid | None
    basis: CompileStaticStr
    # TODO: consider changing CompileStaticIntA to hashable_array?
    periods: CompileStaticIntA | None  # required for gpu-based basis evaluation
    grid_alignment: CompileStaticInt | None  # required for gpu-based grid evaluation
    initial_density_matrix: Array
    occupancies: Array | None = None  # required for gpu-based fock tensor construction

    @property
    def max_number_of_basis_fns(self) -> CompileStaticInt:  # type: ignore
        if self.fock_tensors is not None:
            return len(self.fock_tensors.basis_mask)
        elif self.grid is not None:
            return self.occupancies.shape[-1]  # type: ignore


def preload_system_using_pyscf(
    nuc_pos: ArrayLike,  # nuclei positions FloatAx3  # type: ignore
    atom_z: ArrayLike,  # atomic numbers IntA  # type: ignore
    charge: int,
    spin: int,
    basis: str,
    spin_restricted: bool,
    alignment: Alignment,
    ert_type: ElectRepTensorType | None = None,
    include_fock_tensors: bool = False,
    include_grid: bool = False,
    grid_level: int = 1,
    center: bool = False,
) -> PreloadSystem:
    nuc_pos: Array = onp.array(nuc_pos)
    atom_z: Array = onp.array(atom_z, dtype=onp.uint8)
    order = onp.argsort(atom_z, stable=True)  # e.g. [1, 8, 6, 1, 1] -> [1, 1, 1, 6, 8]
    nuc_pos = nuc_pos[order]
    atom_z = atom_z[order]

    if center:
        nuc_pos -= nuc_pos.mean(axis=0)

    mol = gto.M(
        atom=list(zip(atom_z, nuc_pos * ANGSTROM_TO_BOHR)),
        basis=basis,
        charge=charge,
        spin=spin,
        unit='Bohr',
    )
    n_electrons = int(onp.sum(atom_z) - charge)
    B = mol.nao  # number of basis functions
    if include_fock_tensors:
        assert ert_type is not None, 'ert_type must be provided'
        fock_tensors = preload_fock_tensors_using_pyscf(
            mol, spin, n_electrons, spin_restricted, ert_type, alignment
        )
        occupancies = None
    else:
        fock_tensors = None
        occupancies = compute_electron_occupancy(spin, n_electrons, B, spin_restricted)

    mf = dft.RKS(mol) if spin_restricted else dft.UKS(mol)
    periods, grid, grid_alignment = None, None, None
    if include_grid:
        grid = preload_grid_using_pyscf(mol, mf.grids, grid_level, alignment)
    else:
        grid_alignment = alignment.grid
        periods = z_to_periods(atom_z)

    atom_mask = onp.ones_like(atom_z, dtype=bool)
    initial_density_matrix = mf.get_init_guess()

    if alignment.is_aligned:
        n_atoms = atom_mask.sum()
        if include_grid:
            atom_padding = pad.calc_padding_size(n_atoms, alignment.atom)  # type: ignore
        else:
            atom_padding = 0
            set_periods = set(periods)  # type: ignore
            array_periods = onp.array(periods)
            for period in set_periods:
                align = (
                    alignment.atom
                    if isinstance(alignment.atom, int)
                    else alignment.atom[period]
                )
                n_period = onp.sum(array_periods == period)
                period_padding = pad.calc_padding_size(n_period, align)
                periods += (period,) * period_padding  # type: ignore
                atom_padding += period_padding

        atom_mask = onp.zeros(n_atoms + atom_padding, dtype=bool)
        atom_mask[:n_atoms] = True

        atom_z = onp.pad(atom_z, (0, atom_padding))
        nuc_pos = onp.pad(nuc_pos, ((0, atom_padding), (0, 0)))

        B = initial_density_matrix.shape[-1]
        padding_size = pad.calc_padding_size(B, alignment.basis)
        if spin_restricted:
            initial_density_matrix = onp.pad(
                initial_density_matrix, ((0, padding_size), (0, padding_size))
            )
            if not include_fock_tensors:
                occupancies = onp.pad(occupancies, (0, padding_size))  #  type: ignore
        else:
            initial_density_matrix = onp.pad(
                initial_density_matrix, ((0, 0), (0, padding_size), (0, padding_size))
            )
            if not include_fock_tensors:
                occupancies = onp.pad(occupancies, ((0, 0), (0, padding_size)))  # type: ignore

    return PreloadSystem(
        nuc_pos=nuc_pos.astype(PRECISION.forces),
        atom_z=PerInvHashArray(atom_z.astype(onp.uint8)),
        atom_mask=atom_mask,
        fock_tensors=fock_tensors,
        grid=grid,
        basis=basis,
        periods=periods,
        grid_alignment=grid_alignment,
        initial_density_matrix=initial_density_matrix,
        occupancies=occupancies,
    )
