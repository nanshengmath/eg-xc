"""
Container for molecular structures taken from
H. Helal's and A Fitzgibbon's
"MESS: Modern Electronic Structure Simulations" package
https://arxiv.org/abs/2406.03121
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from periodictable import elements
from pyscf import gto

from egxc.systems.preload import PreloadSystem, PreloadGrid
from egxc.utils.constants import ANGSTROM_TO_BOHR
from egxc.solver.linalg import transformation_matrix

from egxc.utils.typing import (
    Int1,
    IntB,
    Bool2xB,
    IntA,
    BoolA,
    BoolB,
    Float1,
    FloatAx3,
    FloatNx3,
    FloatN,
    FloatNxB,
    FloatNxBx3,
    FloatBxB,
    FloatQxBxB,
    FloatBxBxBxB,
)

from typing import List, Tuple


@dataclass
class FockTensors:
    """
    Tensors that are constant for a given Structure and basis set which are
    used to calculate the Fock matrix.
    """

    basis_mask: BoolB
    overlap: FloatBxB
    core_hamiltonian: FloatBxB
    electron_repulsion_tensor: FloatQxBxB | FloatBxBxBxB
    diagonal_overlap: FloatBxB
    occupancies: IntB | Bool2xB

    @property
    def ert(self) -> FloatQxBxB | FloatBxBxBxB:
        # abbreviation / alias for electron_repulsion_tensor
        return self.electron_repulsion_tensor


@dataclass
class Grid:
    """
    Quadrature grid points, weights for numerical integration and atomic orbitals evaluated at grid points.s
    """

    coords: FloatNx3
    weights: FloatN
    aos: FloatNxB
    grad_aos: FloatNxBx3 | None

    @classmethod
    def create(
        cls, coords: FloatNx3, weights: FloatN, aos: FloatNxB, grad_aos: FloatNxBx3 | None
    ) -> 'Grid':
        mask = weights == 0
        aos = jnp.where(mask[:,None], 1, aos)
        if grad_aos is not None:
            grad_aos = jnp.where(mask[:, None, None], 1, grad_aos)
        return cls(coords, weights, aos, grad_aos)

    @classmethod
    def from_preloaded(cls, pgrid: PreloadGrid) -> 'Grid':
        return cls(
            jnp.asarray(pgrid.coords),
            jnp.asarray(pgrid.weights),
            jnp.asarray(pgrid.aos),
            jnp.asarray(pgrid.grad_aos) if pgrid.grad_aos is not None else None,
        )


@dataclass
class System:
    _nuc_pos: FloatAx3  # TODO: rename to non force nuc_pos ?
    atom_z: IntA  # atomic numbers where Z=255 are masked out
    atom_mask: BoolA
    fock_tensors: FockTensors
    grid: Grid

    @property
    def n_atoms(self) -> int:
        return len(self.atom_z)

    @property
    def n_electrons(self) -> Int1:
        return jnp.sum(self.fock_tensors.occupancies)

    @property
    def atomic_symbol(self) -> List[str]:
        return [elements[z].symbol for z in self.atom_z]

    @classmethod
    def from_preloaded(
        cls,
        psys: PreloadSystem,
        fock_tensors: FockTensors | None = None,
        grid: Grid | None = None,
    ):
        if fock_tensors is None:
            assert psys.fock_tensors is not None, 'Fock tensors must be provided'
            fock_tensors = FockTensors(
                basis_mask=jnp.asarray(psys.fock_tensors.basis_mask),
                overlap=jnp.asarray(psys.fock_tensors.overlap),
                core_hamiltonian=jnp.asarray(psys.fock_tensors.core_hamiltonian),
                electron_repulsion_tensor=jnp.asarray(
                    psys.fock_tensors.electron_repulsion_tensor
                ),
                diagonal_overlap=transformation_matrix(
                    jnp.asarray(psys.fock_tensors.overlap)
                ),
                occupancies=jnp.asarray(psys.fock_tensors.occupancies),
            )

        if grid is None:
            assert psys.grid is not None, 'Grid must be provided'
            grid = Grid.from_preloaded(psys.grid)

        return cls(
            jnp.asarray(psys.nuc_pos),
            atom_z=jnp.asarray(psys.atom_z.array),
            atom_mask=jnp.asarray(psys.atom_mask),
            fock_tensors=fock_tensors if fock_tensors is not None else psys.fock_tensors,
            grid=grid,
        )

    def to_pyscf(self, basis: str) -> gto.Mole:
        """Convert to a PySCF molecule"""
        # manual conversion to bohr, to aid ao test precision by avoiding pySCF's conversion
        atom_z = self.atom_z[self.atom_mask]
        nuc_pos = self._nuc_pos[self.atom_mask]

        occ = self.fock_tensors.occupancies
        if occ.ndim == 1:
            spin = 0
        else:
            spin = occ[0].sum() - occ[1].sum()
        charge = jnp.sum(atom_z) - occ.sum()

        mol = gto.M(
            atom=list(zip(atom_z, nuc_pos * ANGSTROM_TO_BOHR)),
            basis=basis,
            charge=int(charge),
            spin=spin,
            unit='Bohr',
        )
        return mol


def nuclear_energy(nuc_pos: FloatAx3, sys: System) -> Float1:
    """
    Nuclear electrostatic interaction energy.
    Assumes that the input positions are in Angstrom
    """
    idx, jdx = jnp.triu_indices(sys.n_atoms, 1)
    a_a_mask = sys.atom_mask[idx] * sys.atom_mask[jdx]
    u = sys.atom_z[idx] * sys.atom_z[jdx]
    u *= a_a_mask
    rij = nuc_pos[idx, :] - nuc_pos[jdx, :]
    rij *= ANGSTROM_TO_BOHR  # convert distances to Bohr s.t. energies are in Hartree
    return jnp.sum(u * jnp.where(u != 0, 1 / jnp.linalg.norm(rij, axis=1), 0))


def nuclear_energy_and_force(nuc_pos: FloatAx3, sys: System) -> Tuple[Float1, FloatAx3]:
    E, grad = jax.value_and_grad(nuclear_energy)(nuc_pos, sys)
    return E, -grad
