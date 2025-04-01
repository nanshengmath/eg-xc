import jax.numpy as jnp
from typing import Tuple

from egxc.utils.typing import (
    IntE,
    Bool2xE,
    FloatB,
    FloatBxB,
    FloatBxE,
    Float2xBxE,
    PRECISION,
)


def coeff_to_density_matrix(
    coeff: FloatBxE | Float2xBxE, occupancy: IntE | Bool2xE   # in practice E = B
) -> FloatBxB:
    """
    Calculates the density matrix from the coefficient matrix.
    """
    return jnp.einsum('pi,qi,i->pq', coeff, coeff, occupancy)


def transformation_matrix(S: FloatBxB) -> FloatBxB:
    """
    Returns the transformation matrix X that diagonalizes the overlap matrix S.
    """
    # TODO: remove this check?
    assert (
        S.dtype == PRECISION.solver
    ), f'Expected {PRECISION.solver} dtype, got {S.dtype}'

    Lambda, V = jnp.linalg.eigh(S, symmetrize_input=True)
    inv_lambda = jnp.reciprocal(jnp.sqrt(Lambda))
    return jnp.einsum('ab,b,bd->ad', V, inv_lambda, V.T)


def modified_generalized_eigenvalue_problem(
    F: FloatBxB, X: FloatBxB, mask=None
) -> Tuple[FloatB, FloatBxB]:
    """
    Returns a function that solves the generalized eigenvalue problem.
    In the context of SCF calculations F is the Fock matrix and X is the
    transformation matrix which diagonalizes the overlap matrix S.

    TODO: make more robust to degeneracies due to large basis sets
          (see section 7 in https://doi.org/10.3390/molecules25051218)
    """
    F_dash = X.T @ F @ X
    if mask is not None:
        F_dash *= mask
    e, C_dash = jnp.linalg.eigh(F_dash, symmetrize_input=True)
    C = X @ C_dash
    return e, C


# NOTE: jax.scipy.eigh presently (=January 2025) only implements B = None case
# from jax import scipy as jsp
# def direct_generalized_eigenvalue_problem(
#     F: FloatBxB, S: FloatBxB, mask=None
# ) -> Tuple[FloatB, FloatBxB]:
#     """
#     Returns a function that solves the generalized eigenvalue problem.
#     In the context of SCF calculations F is the Fock matrix and S is the overlap matrix.
#     """
#     return jsp.linalg.eigh(F, S)
