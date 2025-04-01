import jax
import jax.numpy as jnp

import flax.linen as nn
import e3nn_jax as e3nn

from egxc.utils.typing import BoolA, FloatN, FloatNx3, FloatAx3, FloatAxN, FloatAxNxRBF
from typing import Callable, Tuple, Literal
from .base import EmbeddingCache

EPSILON = 1e-15


def sin_cos_basis(r: FloatAxN, n_rbf: int) -> FloatAxNxRBF:
    """
    Args:
        r (FloatAxN): distance between atoms in units of cutoff
        n_rbf (int): number of radial basis functions
    Returns:
        FloatAxNxRBF: radial basis functions where RBF = 2 * n_rbf + 1
    """
    r = r[..., None]
    n_rbf = jnp.arange(1, n_rbf + 1, dtype=r.dtype)  # type: ignore
    out = jnp.sqrt(2.0) * jnp.concatenate(
        [
            jnp.ones_like(r) * 0.1,  # TODO: why constant 0.1?
            jnp.sin(n_rbf * jnp.pi * r),
            jnp.cos(n_rbf * jnp.pi * r),
        ],
        axis=-1,
    )
    return out


def radial_embedding_basis(r: jax.Array, n: int):
    vec_poly_envelope: Callable[[jax.Array], jax.Array] = jnp.vectorize(
        e3nn.poly_envelope(5, 2)  # TODO: why 5, 2?
    )  # https://mariogeiger.ch/polynomial_envelope_for_gnn.pdf
    return sin_cos_basis(r, n) * vec_poly_envelope(r)[..., None]


class Encoder(nn.Module):
    """
    Module that encodes the electron density on to a nuclei-centered point cloud.
    """

    irreps: e3nn.Irreps
    cutoff: float  # TODO: check units, NOTE: This cutoff is independent of the GNN cutoff
    num_radial_filters: int = 16  # RBF dimension
    nuclei_partitioning: Literal[None, 'Exponential', 'Gaussian'] = None
    _quadrature_points_per_atom_scaling: int = 8

    def setup(self) -> None:
        if self.nuclei_partitioning is not None:
            self.sigma = self.param(  # TODO: should this be a learnable parameter?
                'partitioning_smoothness',
                jax.nn.initializers.ones,
                (),
                jnp.float32,
            )

            def density_partitioning(dist: FloatAxN, atom_mask: BoolA) -> FloatAxN:
                if self.nuclei_partitioning == 'Gaussian':
                    claim = jnp.exp(-0.5 * dist**2 / (self.sigma**2))
                else:
                    # exponential
                    claim = jnp.exp(-dist / jnp.abs(self.sigma))
                claim = claim * atom_mask[:, None]  # mask out fake (padding) atoms
                normalizer = claim.sum(0, keepdims=True)
                share = claim / (normalizer + EPSILON)
                return share

            self.density_partitioning_fn = density_partitioning

    def __call__(
        self,
        nuc_pos: FloatAx3,
        atom_mask: BoolA,
        grid_coords: FloatNx3,
        weights: FloatN,
        n: FloatN,
    ) -> Tuple[e3nn.IrrepsArray, EmbeddingCache]:
        """
        TODO: Should we include other features like s, xi, tau in the embedding?
        """
        # change to units of cutoff
        grid_coords = grid_coords / self.cutoff
        nuc_pos = nuc_pos / self.cutoff
        # compute distances
        diff = nuc_pos[:, None] - grid_coords[None]  # FloatAxNx3
        dist = jnp.linalg.norm(  # distance between nuclei and grid points
            diff, axis=-1
        )  # FloatAxN  # TODO: why did nicholas write a safe_norm function?
        # optimize calculations by using the distance based cutoff
        truncated_idx = jnp.argsort(dist, axis=1)
        A, N = dist.shape
        T = self._quadrature_points_per_atom_scaling * N // A
        # take at least T points but at most the entire grid
        truncated_idx = truncated_idx[:, : min(N, T)]
        diff = jnp.take_along_axis(diff, truncated_idx[:, :, None], axis=1)
        dist = jnp.take_along_axis(dist, truncated_idx, axis=1)
        n = n[truncated_idx]
        weights = weights[truncated_idx]

        radial_basis_vals = radial_embedding_basis(  # FloatAxNxRBF
            dist, self.num_radial_filters
        )
        # apply nuclei wise partitioning of the quadrature points
        if self.nuclei_partitioning is not None:
            partitioning = self.density_partitioning_fn(dist, atom_mask)
            radial_basis_vals *= partitioning[..., None]
        directions = diff / (dist[..., None] + EPSILON)  # FloatAxNx3
        spherical_harmonics = e3nn.spherical_harmonics(
            self.irreps, directions, normalize=True, normalization='norm'
        )
        node_feats = jnp.einsum(
            'atr,ath,at,at->arh',
            radial_basis_vals,
            spherical_harmonics.array,
            n,
            weights,
        )  # FloatAxRBFxH
        node_feats = e3nn.IrrepsArray(self.irreps, node_feats)

        return node_feats, (N, truncated_idx, spherical_harmonics, radial_basis_vals)
