"""
Contains functions to calculate the Fock matrix
for a given density matrix using these tensors.

For a general understanding of this module see section 3 and 6 in:
Susi Lehtola et al. "An Overview of Self-Consistent Field Calculations Within Finite Basis Sets"
Molecules 2020, 25 (5), 1218. https://doi.org/10.3390/molecules25051218.
"""

import jax.numpy as jnp
import flax.linen as nn
import einops

from egxc.xc_energy import XCModule
from egxc.systems.base import System

from egxc.utils.typing import (
    Float1,
    FloatAx3,
    FloatBxB,
    Float2xBxB,
    FloatQxBxB,
    FloatBxBxBxB,
    ElectRepTensorType,
)
from typing import Tuple, Dict


class FockMatrix(nn.Module):
    xc_module: XCModule
    ert_type: ElectRepTensorType
    spin_restricted: bool
    # TODO: check correct return Precision

    def setup(self):
        def compute_coulomb_matrix(
            density_matrix: FloatBxB | Float2xBxB,
            electron_repulsion_tensor: FloatQxBxB | FloatBxBxBxB,
        ) -> FloatBxB:
            P = density_matrix if self.spin_restricted else density_matrix.sum(axis=0)
            if self.ert_type == ElectRepTensorType.EXACT:
                J = jnp.einsum('ijkl,ij->kl', electron_repulsion_tensor, P)
            elif self.ert_type == ElectRepTensorType.DENSITY_FITTED:
                J = jnp.einsum(
                    'Pij,Pkl,ij->kl',
                    electron_repulsion_tensor,
                    electron_repulsion_tensor,
                    P,
                )
            else:
                raise ValueError(f'Invalid ert_type: {self.ert_type}')
            return J

        self.coulomb_matrix_fn = compute_coulomb_matrix

        def preprocessing(
            nuc_pos: FloatAx3,
            sys: System,
        ) -> Tuple[FloatBxB | Float2xBxB, Dict]:
            non_local_kwargs = {}
            if self.xc_module.xc_functional.is_hybrid:
                non_local_kwargs['eri_tensor'] = sys.fock_tensors.ert
            if self.xc_module.xc_functional.is_graph_based:
                non_local_kwargs['atom_mask'] = sys.atom_mask
                non_local_kwargs['nuc_pos'] = nuc_pos
                non_local_kwargs['grid_coords'] = sys.grid.coords
            H_core = sys.fock_tensors.core_hamiltonian
            if not self.spin_restricted:
                H_core = einops.repeat(H_core, 'i j -> spin i j', spin=2)
            return H_core, non_local_kwargs

        self.preprocessing = preprocessing

    def __call__(
        self,
        nuc_pos: FloatAx3,
        density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> FloatBxB | Float2xBxB:
        return self.fock_matrix(nuc_pos, density_matrix, sys)

    def fock_matrix(
        self,
        nuc_pos: FloatAx3,
        density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> FloatBxB | Float2xBxB:
        """
        Calculates the Fock matrix for a given coefficient matrix.
        """
        P = density_matrix
        H_core, non_local_kwargs = self.preprocessing(nuc_pos, sys)
        J = self.coulomb_matrix_fn(P, sys.fock_tensors.ert)

        V_xc = self.xc_module.xc_potential(
            P, sys.grid, sys.fock_tensors.basis_mask, **non_local_kwargs
        )
        return H_core + J + V_xc

    def energy(
        self,
        nuc_pos: FloatAx3,
        density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> Tuple[Float1, Float1]:
        """
        returns the energies due to (core hamiltonian + coulomb, exchange-correlation)
        """
        P = density_matrix
        H_core, non_local_kwargs = self.preprocessing(nuc_pos, sys)
        J = self.coulomb_matrix_fn(P, sys.fock_tensors.ert)
        e_xc = self.xc_module.xc_energy(P, sys.grid, **non_local_kwargs)
        return ((H_core + 0.5 * J) * P).sum(), e_xc

    def energy_and_fock_matrix(
        self,
        nuc_pos: FloatAx3,
        density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> Tuple[Tuple[Float1, Float1], FloatBxB | Float2xBxB]:
        P = density_matrix
        H_core, non_local_kwargs = self.preprocessing(nuc_pos, sys)
        J = self.coulomb_matrix_fn(P, sys.fock_tensors.ert)
        e_xc, v_xc = self.xc_module.xc_energy_and_potential(
            P, sys.grid, sys.fock_tensors.basis_mask, **non_local_kwargs
        )
        F = H_core + J + v_xc
        return (((H_core + 0.5 * J) * P).sum(), e_xc), F
