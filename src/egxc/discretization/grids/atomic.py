"""
This is adapted from the PySCF library
https://github.com/pyscf/pyscf/blob/master/pyscf/dft/gen_grid.py
"""

import numpy as onp
import jax.numpy as jnp

from egxc.discretization.grids import lebedev
from egxc.utils.typing import FloatN, FloatNx3
from egxc.utils.constants import (
    BRAGG_RADII,
    PSE_ROW_START,
    RAD_GRIDS,
    ANG_ORDER,
    TREUTLER_AHLRICHS_XI,
)

from typing import Dict, Tuple, Set, Callable
from numpy.typing import ArrayLike

FloatR = onp.ndarray


def _nwchem_prune(
    atomic_number: int, radial_coords: ArrayLike, max_angular_grids: int
) -> ArrayLike:
    """
    Prunes the number of angular grids for NWChem based on radial coordinates and atomic properties.

    Args:
        atomic_number (int): The nuclear charge of the atom (Z).
        radial_coords (ArrayLike): 1D array of radial grid coordinates (in Bohr).
        max_angular_grids (int): Maximum number of angular grid points.

    Returns:
        ArrayLike: A 1D array of angular grid counts corresponding to each radial grid point.
    """
    # Threshold factors for pruning based on atomic number ranges
    thresholds = onp.array(
        [(0.25, 0.5, 1.0, 4.5), (0.1667, 0.5, 0.9, 3.5), (0.1, 0.4, 0.8, 2.5)]
    )
    lebedev_grids = lebedev.LEBEDEV_NGRID[4:]  # Skip smaller grid sizes [38, 50, 74, ...]

    # Special case: Fixed angular grids for smaller grid sizes
    if max_angular_grids < 50:
        return max_angular_grids * onp.ones_like(radial_coords)
    elif max_angular_grids == 50:
        lebedev_levels = onp.array([1, 2, 2, 2, 1])
    else:
        # Find Lebedev level corresponding to max_angular_grids
        grid_index = onp.where(lebedev_grids == max_angular_grids)[0][0]
        lebedev_levels = onp.array([1, 3, grid_index - 1, grid_index, grid_index - 1])

    # Determine pruning thresholds based on the atomic number
    atomic_radius = BRAGG_RADII[atomic_number] + 1e-200  # Avoid division by zero
    if atomic_number <= 2:  # Hydrogen, Helium
        threshold = thresholds[0]
    elif atomic_number <= 10:  # Lithium to Neon
        threshold = thresholds[1]
    else:  # Other elements
        threshold = thresholds[2]

    # Determine pruning level for each radial coordinate
    levels = ((radial_coords / atomic_radius).reshape(-1, 1) > threshold).sum(axis=1)
    angular_grids = lebedev_levels[levels]
    return lebedev_grids[angular_grids]


def _gauss_chebyshev(n: int) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Generates the Gauss-Chebyshev quadrature grid for the interval [0, infinity).

    This method computes the integration nodes and weights based on the
    scheme described by Pérez-Jordá et al. in:
    A Simple, Reliable and Efficient Scheme for Automatic Numerical Integration.
    Computer Physics Communications 1992, 70 (2), 271–284.
    https://doi.org/10.1016/0010-4655(92)90192-2.

    n(int): Number of quadrature points.
    Returns: quadrature nodes, quadrature weights
    """
    ln2 = 1 / onp.log(2)  # Precompute natural log of 2
    # Compute Chebyshev nodes
    x = onp.arange(1, n + 1) * onp.pi / (n + 1)
    # Compute intermediate values for the transformation
    zeta = (n - 1 - onp.arange(n) * 2) / (n + 1.0) + (
        1 + 2.0 / 3 * onp.sin(x) ** 2
    ) * onp.sin(2 * x) / onp.pi
    # Symmetrize and transform
    zeta = (zeta - zeta[::-1]) / 2
    r = 1 - onp.log1p(zeta) * ln2  # Nodes
    # Compute weights
    weights = 16.0 / (3 * (n + 1)) * onp.sin(x) ** 4 * ln2 / (1 + zeta)
    return r, weights


def _treutler_ahlrichs(n, Z: int) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Treutler-Ahlrichs [JCP 102, 346 (1995); DOI:10.1063/1.469408] (M4) radial grids
    """
    xi = TREUTLER_AHLRICHS_XI[Z]
    r = onp.empty(n)
    dr = onp.empty(n)
    step = onp.pi / (n + 1)
    ln2 = xi / onp.log(2)
    for i in range(n):
        x = onp.cos((i + 1) * step)
        r[i] = -ln2 * (1 + x) ** 0.6 * onp.log((1 - x) / 2)
        dr[i] = (
            step
            * onp.sin((i + 1) * step)
            * ln2
            * (1 + x) ** 0.6
            * (-0.6 / (1 + x) * onp.log((1 - x) / 2) + 1 / (1 - x))
        )
    return r[::-1], dr[::-1]


def generate(
    elements: Set[int], level: int, prune: Callable = _nwchem_prune
) -> Dict[int, Tuple[FloatNx3, FloatN]]:
    """Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the nuclear charge z for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    """

    def prange(start, end, step):
        # TODO: do I need this?
        """This function splits the number sequence between "start" and "end"
        using uniform "step" length. It yields the boundary (start, end) for each
        fragment.
        """
        if start < end:
            for i in range(start, end, step):
                yield i, min(i + step, end)

    atom_grids_tab: Dict[int, Tuple[FloatNx3, FloatN]] = {}
    for z in set(elements):
        if z == 0:
            continue
        period = (z > PSE_ROW_START).sum()
        n_rad = RAD_GRIDS[level, period]
        n_ang = lebedev.LEBEDEV_ORDER[ANG_ORDER[level, period]]  # type: ignore
        rad, dr = _treutler_ahlrichs(n_rad, z)  # type: ignore

        rad_weight = 4 * onp.pi * rad**2 * dr

        if callable(prune):
            angs = prune(z, rad, n_ang)
        else:
            angs = [n_ang] * n_rad

        angs = onp.array(angs)
        coords = []
        weight = []
        for n in sorted(set(angs)):
            grid = lebedev.MakeAngularGrid(n)
            idx = onp.where(angs == n)[0]
            for i0, i1 in prange(0, len(idx), 12):  # 12 radi-grids as a group
                coords.append(
                    onp.einsum('i,jk->jik', rad[idx[i0:i1]], grid[:, :3]).reshape(-1, 3)
                )
                weight.append(
                    onp.einsum('i,j->ji', rad_weight[idx[i0:i1]], grid[:, 3]).ravel()
                )
        atom_grids_tab[int(z)] = (jnp.vstack(coords), jnp.hstack(weight))
    return atom_grids_tab
