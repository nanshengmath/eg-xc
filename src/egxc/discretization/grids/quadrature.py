"""
Differentiable grids for numerical integration in quantum chemistry.

This is adapted from the PySCF library
https://github.com/pyscf/pyscf/blob/master/pyscf/dft/gen_grid.py
"""

import jax
import jax.numpy as jnp
import numpy as onp
import einops
from functools import partial

from egxc.discretization.grids import atomic
from egxc.utils import constants, pad
from egxc.utils.constants import BRAGG_RADII

from typing import Tuple, Callable, List, Set
from egxc.utils.typing import (
    PRECISION,
    BoolA,
    IntA,
    FloatA,
    FloatBxB,
    FloatAxN,
    FloatAx3,
    FloatAxA,
    FloatNx3,
    FloatN,
    PermutiationInvariantHashableArray as PerInvHashArray,
)

RadiiAdjustFn = Callable[[FloatA, int, int, jax.Array], jax.Array]
QuadratureGridFn = Callable[[FloatAx3, PerInvHashArray, FloatA, BoolA], Tuple[FloatNx3, FloatN]]


def treutler_atomic_radii_adjust(
    atomic_radii: FloatA, i: int, j: int, partitioning: FloatBxB
) -> FloatBxB:
    """
    Adjust atomic radii using the Treutler scheme to account for atomic partitioning.
    """
    adjusted_radii = jnp.sqrt(atomic_radii) + 1e-200
    radii_ratio = adjusted_radii.reshape(-1, 1) * (1.0 / adjusted_radii)
    adjustment_factor = 0.25 * (radii_ratio.T - radii_ratio)
    adjustment_factor = jnp.where(adjustment_factor < -0.5, -0.5, adjustment_factor)
    adjustment_factor = jnp.where(adjustment_factor > 0.5, 0.5, adjustment_factor)
    return (1.0 - partitioning**2) * adjustment_factor[i, j] + partitioning


def compute_interatomic_distances(nuc_pos: FloatAx3, atom_mask: BoolA) -> FloatAxA:
    """
    Compute interatomic distances for a set of atomic coordinates.
    """
    distances = jnp.linalg.norm(
        nuc_pos.reshape(-1, 1, 3) - nuc_pos, axis=2
    )
    # Explicitly set diagonal elements to zero to avoid numerical errors
    distances = distances.at[jnp.diag_indices(distances.shape[0])].set(0.0)
    # Add a small epsilon to distances between padded atoms
    dist_eps = jnp.outer(atom_mask, ~atom_mask)
    dist_eps += dist_eps.T
    dist_eps += jnp.outer(~atom_mask, ~atom_mask)
    return distances + 1e-9 * dist_eps


def becke_smoothing(partitioning: FloatN) -> FloatN:
    """
    Apply the Becke smoothing function to the partitioning variable.
    """
    for _ in range(3):
        partitioning = (3 - partitioning**2) * partitioning * 0.5
    return partitioning


def get_grid_fn(
    level: int,
    elements: Set[int] | List[int] | IntA | onp.ndarray,
    alignment: int,
    radii_method: RadiiAdjustFn = treutler_atomic_radii_adjust,
    smoothing_function: Callable = becke_smoothing,
) -> QuadratureGridFn:
    if isinstance(elements, (onp.ndarray, jnp.ndarray)):
        elements = set(elements.tolist())  # type: ignore
    elif isinstance(elements, list):
        elements = set(elements)
    atomic_grids = atomic.generate(elements, level)  # type: ignore

    @partial(jax.jit, static_argnames=('atom_z'))  # TODO: could also do this by period
    def grid_fn(nuc_pos: FloatAx3, atom_z: PerInvHashArray, atom_mask: BoolA) -> Tuple[FloatNx3, FloatN]:
        nuclei_positions = nuc_pos * constants.ANGSTROM_TO_BOHR  # TODO: unit conversion
        atom_radii = jnp.array(
            [BRAGG_RADII[z] for z in atom_z], dtype=PRECISION.quadrature
        )
        A = len(atom_z)

        interatomic_distances = compute_interatomic_distances(nuclei_positions, atom_mask)

        def compute_becke_weights(grid_coords: FloatNx3) -> FloatAxN:
            N = grid_coords.shape[0]
            displacement = grid_coords[None] - nuclei_positions[:, None]
            grid_distances = jnp.sqrt(
                jnp.einsum('ijk,ijk->ij', displacement, displacement)
            )
            becke_weights = jnp.ones((A, N))

            atom_indices_1, atom_indices_2 = jnp.tril_indices(A, k=-1)

            def compute_partitioning(i, j):
                partitioning = (1 / interatomic_distances[i, j]) * (
                    grid_distances[i] - grid_distances[j]
                )
                partitioning = radii_method(atom_radii, i, j, partitioning)
                return smoothing_function(partitioning)

            partitionings = jax.vmap(compute_partitioning)(atom_indices_1, atom_indices_2)
            partitionings = partitionings * atom_mask[atom_indices_1, None] * atom_mask[atom_indices_2, None]
            partitionings += ~atom_mask[atom_indices_1, None] * atom_mask[atom_indices_2, None]
            becke_weights = becke_weights.at[atom_indices_1].mul(
                0.5 * (1.0 - partitionings)
            )
            becke_weights = becke_weights.at[atom_indices_2].mul(
                0.5 * (1.0 + partitionings)
            )
            return becke_weights

        coords = []
        weights = []
        for z in elements:
            if z == 0:
                continue
            z_idx = atom_z.array == z
            if onp.any(z_idx):
                a_grid, a_weights = atomic_grids[z]
                out_grid = nuclei_positions[z_idx, None] + a_grid[None]  # shape (Z, N, 3)
                becke_weights = jax.vmap(compute_becke_weights)(out_grid)  # shape (Z, A, N)
                norm = jnp.sum(becke_weights, axis=1, keepdims=True)
                becke_weights = becke_weights[:, z_idx] / norm # shape (Z, Z, N)
                out_weights = einops.einsum(a_weights, becke_weights, 'N, Z Z N -> Z N')
                coords.append(out_grid.reshape(-1, 3))
                weights.append(out_weights.flatten())

        coords = jnp.vstack(coords)
        weights = jnp.hstack(weights)

        return pad.pad_quadrature_grid(alignment, coords, weights)

    return grid_fn
