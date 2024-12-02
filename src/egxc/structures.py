"""
Container for molecular structures taken from
H. Helal's and A Fitzgibbon's
"MESS: Modern Electronic Structure Simulations" package
https://arxiv.org/abs/2406.03121
"""

from typing import List

import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
from jax import value_and_grad
from jaxtyping import Scalar
from numpy.typing import ArrayLike
from periodictable import elements
from pyscf import gto

from .utils.typing import BoolN, FloatNx3, IntN


@dataclass
class Structure:
    atomic_numbers: IntN
    positions: FloatNx3
    charge: int
    spin: int

    def __post_init__(self):
        # single atom case
        assert jnp.ndim(self.atomic_numbers) == 1, 'Atomic numbers must be 1D'
        assert jnp.ndim(self.positions) == 2, 'Positions must be 2D'
        # check that atoms are ordered by charge
        z = self.atomic_numbers
        assert jnp.all(z[:-1] <= z[1:]), 'Atomic numbers are not ordered by charge'

    @property
    def periods(self) -> IntN:
        Z = self.atomic_numbers
        p = jnp.zeros_like(Z)
        # fmt: off
        p = jnp.where(     Z <=  2, 1, p)
        p = jnp.where( 2 < Z <= 10, 2, p)
        p = jnp.where(10 < Z <= 18, 3, p)
        p = jnp.where(18 < Z <= 36, 4, p)
        p = jnp.where(36 < Z <= 54, 5, p)
        p = jnp.where(54 < Z <= 86, 6, p)
        p = jnp.where(86 < Z,       7, p)
        # fmt: on
        return p

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def n_electrons(self) -> int:
        return onp.sum(self.atomic_numbers) - self.charge

    @property
    def atomic_sym(self) -> List[str]:
        return [elements[z].symbol for z in self.atomic_numbers]

    def pad_structure(self, n: int) -> 'PaddedStructure':
        """Pad the structure with n zeros"""
        assert n >= self.n_atoms, 'Cannot pad with fewer atoms'
        temp = jnp.arange(n)
        is_atom = temp < self.n_atoms
        padded = PaddedStructure(
            atomic_numbers=jnp.zeros(self.n_atoms + n),
            positions=jnp.zeros((self.n_atoms + n, 3)),
            charge=self.charge,
            spin=self.spin,
            mask=is_atom,
        )
        return padded

    def to_pyscf_mol(self, basis: str) -> gto.Mole:
        """Convert to a PySCF molecule"""
        mol = gto.M(
            atom=list(zip(self.atomic_numbers, self.positions)),
            basis=basis,
            charge=self.charge,
            spin=self.spin,
        )
        return mol


@dataclass
class PaddedStructure(Structure):
    mask: BoolN

    def to_pyscf_mol(self, basis: str) -> gto.Mole:
        mol = gto.M(
            atom=list(zip(self.atomic_numbers[self.mask], self.positions[self.mask])),
            basis=basis,
            charge=self.charge,
            spin=self.spin,
        )
        return mol


def create_structure(
    Z: ArrayLike, pos: ArrayLike, charge: int = 0, spin: int = 0
) -> Structure:
    """Convenience function to create a Structure"""
    Z = jnp.atleast_1d(Z).astype(int)  # type: ignore
    pos = jnp.atleast_2d(pos)  # type: ignore
    return Structure(atomic_numbers=Z, positions=pos, charge=charge, spin=spin)


def nuclear_energy(structure: Structure) -> Scalar:
    """Nuclear electrostatic interaction energy"""
    idx, jdx = jnp.triu_indices(structure.n_atoms, 1)
    u = structure.atomic_numbers[idx] * structure.atomic_numbers[jdx]
    rij = structure.positions[idx, :] - structure.positions[jdx, :]
    return jnp.sum(u / jnp.linalg.norm(rij, axis=1))


def nuclear_energy_and_force(structure: Structure):
    @value_and_grad
    def energy_and_grad(pos: FloatNx3, structure: Structure):
        idx, jdx = jnp.triu_indices(structure.n_atoms, 1)
        u = structure.atomic_numbers[idx] * structure.atomic_numbers[jdx]
        rij = pos[idx, :] - pos[jdx, :]
        return jnp.sum(u / jnp.linalg.norm(rij, axis=1))

    E, grad = energy_and_grad(structure.positions, structure)
    return E, -grad.position
