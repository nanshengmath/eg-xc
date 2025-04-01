import jax.numpy as jnp
from flax.struct import dataclass

from egxc.systems.base import FockTensors
from egxc.utils.typing import FloatBxB, FloatSCF, FloatSCFxSCF, FloatSCFxBxB
from typing import Tuple


def compute_residual(
    F: FloatBxB, P: FloatBxB, fock_tensors: FockTensors
) -> FloatBxB:
    """
    Computes the residual matrix for the Fock matrix.
    F: Fock matrix
    P: Density matrix
    cst: constant system tensors containing the overlap matrix
    """
    temp = jnp.einsum('ab,bc,cd->ad', F, P, fock_tensors.overlap)
    res = fock_tensors.diagonal_overlap.T @ (temp - temp.T) @ fock_tensors.diagonal_overlap
    res = (res - res.T) / 2  # Recover anti-symmetry violated by numerical errors
    return res


def solve_pulay_equation(overlap: FloatSCFxSCF, current_cycle: int) -> FloatSCF:
    B = overlap
    total_cycles = overlap.shape[0]
    constraint_idx = current_cycle + 1
    set_vec = -1 * (jnp.arange(total_cycles) < constraint_idx)
    B = B.at[:, constraint_idx].set(set_vec)
    B = B.at[constraint_idx, :].set(set_vec)
    B = B.at[constraint_idx, constraint_idx].set(0)
    rhs = jnp.zeros(total_cycles).at[constraint_idx].set(-1)
    fock_coeffs = jnp.linalg.solve(B, rhs)  # (x0, ..., x_{n-1}, lambda, 0, ..., 0)
    return fock_coeffs  # TODO: test fock_coeffs[current_cycle + 1:] == 0


@dataclass
class DiisState:
    overlap: FloatBxB  # do not confuse with the basis set overlap matrix
    fock_trajectory: FloatSCFxBxB
    res_trajectory: FloatSCFxBxB

    @classmethod
    def init(
        cls,
        total_cycles: int,
        fock_matrix: FloatBxB,
        density_matrix: FloatBxB,
        fock_tensors: FockTensors,
    ):
        N_bas = fock_matrix.shape[0]
        overlap = jnp.eye(total_cycles + 1)
        fock_trajectory = jnp.zeros((total_cycles, N_bas, N_bas)).at[0].set(fock_matrix)
        residual = compute_residual(fock_matrix, density_matrix, fock_tensors)
        res_trajectory = jnp.zeros((total_cycles, N_bas, N_bas)).at[0].set(residual)
        return cls(overlap, fock_trajectory, res_trajectory)


def diis_update(
    current_cycle: int,
    raw_fock_matrix: FloatBxB,
    state: DiisState,
    density_matrix: FloatBxB,
    fock_tensors: FockTensors
) -> Tuple[FloatBxB, DiisState]:
    """
    Direct Inversion of the Iterative Subspace (DIIS) to accelerate the
    convergence of the Self-Consistent Field (SCF) method.
    Returns the DIIS update to the Fock matrix.

    current_cycle: current cycle of the SCF method
    raw_fock_matrix: standard Fock matrix
    density_matrix: density matrix
    state: DIIS state based on the previous cycles
    cst: constant system tensors containing the overlap matrix

    returns:
        (Fock matrix updated by DIIS, updated DIIS state)

    Implementation inspired by
    https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3b_rhf-diis.ipynb
    but adapted to be jax compile friendly.
    """
    residual = compute_residual(raw_fock_matrix, density_matrix, fock_tensors)
    i = current_cycle
    res_trajectory = state.res_trajectory.at[i].set(residual)
    new_overlap = jnp.einsum('ikl,kl->i', res_trajectory, residual)
    overlap = state.overlap.at[i,:-1].set(new_overlap)
    overlap = overlap.at[:-1, i].set(new_overlap)
    fock_coeffs = solve_pulay_equation(overlap, i)
    fock_trajectory = state.fock_trajectory.at[i].set(raw_fock_matrix)
    F_out = jnp.einsum('i,ijk->jk', fock_coeffs[:-1], fock_trajectory)
    F_out = jnp.where(
        jnp.isnan(F_out).any(), raw_fock_matrix, F_out
    )  # this is necessary, since B becomes singular once it converges converged
    return F_out, DiisState(overlap, fock_trajectory, res_trajectory)
