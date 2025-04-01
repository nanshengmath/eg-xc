import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn

from egxc.xc_energy.functionals.learnable.nn.base import MLP

from egxc.utils.typing import (
    Float1,
    BoolA,
    FloatAx3,
    FloatAxA,
    FloatAxAx3,
    FloatAxF,
    FloatAxFx3,
    FloatAxAx3F,
)
from typing import Tuple, Callable


def cosine_cutoff(x: jax.Array, cutoff: float) -> jax.Array:
    """Behler-style cosine cutoff function"""
    x = 0.5 * (jnp.cos((jnp.pi / cutoff) * x) + 1)
    return jnp.where(x < cutoff, x, 0)


class ScalarFilter(nn.Module):
    cutoff_dist: float
    atom_features: int
    radial_basis_fn: int

    def setup(self) -> None:
        n = jnp.arange(1, self.radial_basis_fn + 1)
        self.prefactors = n * jnp.pi / self.cutoff_dist

    @nn.compact
    def __call__(self, x: FloatAxA) -> FloatAxAx3F:
        """
        input: interatomic distances shape (N_atoms, N_atoms)
        output: scalar filter shape (N_atoms, N_atoms, 3 * n_features)
        """
        x = jnp.sin(
            x[..., None] * self.prefactors[None, None, :]
        )  # shape (N_atoms, N_atoms, n_basis)
        x = nn.Dense(3 * self.atom_features)(x)
        return x


class MessageBlock(nn.Module):
    atom_features: int
    cutoff_dist: float
    radial_basis_fn: int

    @nn.compact
    def __call__(
        self, s: FloatAxF, v: FloatAxFx3, dr: FloatAxAx3, atom_mask: BoolA
    ) -> Tuple[FloatAxF, FloatAxFx3]:
        """
        Computes messages between atoms.
        Feature-wise updates across atoms.

        Args:
            s: scalar atom features
            v: equivariant atom features
            dr: distance vectors between atoms
            atom_mask: padding mask for atoms

        Returns:
            ds_msg: scalar messages
            dv_msg: equivariant messages
        """
        distances = jnp.linalg.norm(dr, axis=-1)

        phi = nn.Dense(self.atom_features)(s)
        phi = nn.silu(phi)
        phi = nn.Dense(3 * self.atom_features)(phi)
        f_cut = cosine_cutoff(distances, self.cutoff_dist)  # shape (N_atoms, N_atoms)
        W = (
            ScalarFilter(self.cutoff_dist, self.atom_features, self.radial_basis_fn)(
                distances
            )
            * f_cut[..., None]
        )
        msg = jnp.einsum(
            'jf, ijf, j -> ijf', phi, W, atom_mask
        )  # shape (N_atoms, N_atoms, 3 * n_features)

        # scalar messages
        ds_msg = msg[..., : self.atom_features].sum(axis=1)

        # equivariant messages
        msg_vv = msg[..., self.atom_features : 2 * self.atom_features]
        msg_vs = msg[..., 2 * self.atom_features :]
        e_r_save = dr / (distances + 1e-9)[..., None]
        dv_msg = jnp.einsum('jfv,  ijf -> ifv', v, msg_vv) + jnp.einsum(
            'ijv,  ijf -> ifv', e_r_save, msg_vs
        )
        return ds_msg, dv_msg


class UpdateBlock(nn.Module):
    atom_features: int

    @nn.compact
    def __call__(self, s: FloatAxF, v: FloatAxFx3) -> Tuple[FloatAxF, FloatAxFx3]:
        """
        Updates the atom features.
        Atom-wise updates across features.

        Args:
            s: scalar atom features
            v: equivariant atom features

        Returns:
            ds_up: scalar updates
            dv_up: equivariant updates
        """
        # learnable linear combinations of equivariant vectors
        Vv = nn.Einsum(
            (self.atom_features, self.atom_features), 'ifv, gf -> igv', use_bias=False
        )(v)  # shape (N_atoms, n_features, 3)
        Uv = nn.Einsum(
            (self.atom_features, self.atom_features), 'ifv, gf -> igv', use_bias=False
        )(v)  # shape (N_atoms, n_features, 3)
        norm_Vv = jnp.linalg.norm(Vv, axis=-1)
        scalar_prod_Vv_Uv = (Vv * Uv).sum(axis=-1)

        scalar_features = jnp.concatenate(
            [s, norm_Vv], axis=-1
        )  # shape (N_atoms, 2 * n_features)
        atm_rep = nn.Dense(self.atom_features)(scalar_features)
        atm_rep = nn.silu(atm_rep)
        atm_rep = nn.Dense(3 * self.atom_features)(
            atm_rep
        )  # shape (N_atoms, 3 * n_features)

        ds_up = (
            atm_rep[..., : self.atom_features]
            + atm_rep[..., self.atom_features : 2 * self.atom_features]
            * scalar_prod_Vv_Uv
        )
        dv_up = atm_rep[..., 2 * self.atom_features :, None] * Uv
        return ds_up, dv_up


class PaiNN(nn.Module):
    """
    Implementation of polarizable atom interaction neural network by Schütt et al.
    https://doi.org/10.48550/arXiv.2102.03150

    TODO: check if layer norm is needed
    TODO: Can irreps conversions be avoided?
    """

    atom_feature_dim: int  # F: number of features per atom
    cutoff: float
    layers: int  # L: number of message passing layers
    radial_basis_fn: int = 20  # number of radial basis functions
    irreps_out: e3nn.Irreps = e3nn.Irreps('0e + 1o')
    _readout_activation: Callable[[jax.Array], jax.Array] = nn.silu

    @nn.compact
    def __call__(
        self,
        node_features: e3nn.IrrepsArray,  # (A, RBF, (l,m))  with m,l as in Y_{l,m}
        nuc_pos: FloatAx3,
        atom_mask: BoolA,
    ) -> Tuple[Float1, e3nn.IrrepsArray]:  # (A, F, (l,m))  with m,l as in Y_{l,m}
        s, v = self.convert_irreps_to_scalar_and_vector(node_features)
        # preprocessing
        dr = nuc_pos[:, None, :] - nuc_pos[None, :, :]
        s = nn.LayerNorm()(s)
        v_norms = jnp.linalg.norm(v, axis=-1)
        v = v * (nn.LayerNorm()(v_norms) / (v_norms + 1e-9))[..., None]
        # apply message passing and node updates
        for _ in range(self.layers):
            ds_msg, dv_msg = MessageBlock(
                self.atom_feature_dim, self.cutoff, self.radial_basis_fn
            )(s, v, dr, atom_mask)

            s += ds_msg
            s = nn.LayerNorm()(s)
            v += dv_msg
            v_norms = jnp.linalg.norm(v, axis=-1)
            v = v * (nn.LayerNorm()(v_norms) / (v_norms + 1e-9))[..., None]

            ds_up, dv_up = UpdateBlock(self.atom_feature_dim)(s, v)
            s += ds_up
            s = nn.LayerNorm()(s)
            v += dv_up
            v_norms = jnp.linalg.norm(v, axis=-1)
            v = v * (nn.LayerNorm()(v_norms) / (v_norms + 1e-9))[..., None]

        # optional graph readout
        readout = MLP(
            [self.atom_feature_dim, self.atom_feature_dim, 1],
            activation=self._readout_activation,
            init_last_layer_to_zero=True,
        )(s).sum()

        # convert back to e3nn format
        node_features = self.convert_to_irreps(s, v)

        return readout, node_features.mul_to_axis()  # scalar, (A, F, (l,m))  with m,l as in Y_{l,m}

    @nn.compact
    def convert_irreps_to_scalar_and_vector(
        self, node_features: e3nn.IrrepsArray
    ) -> Tuple[FloatAxF, FloatAxFx3]:
        irreps = self.atom_feature_dim * e3nn.Irreps('0e+1o')  # = Fx0e + Fx1o
        node_features = node_features.axis_to_mul()  # (A, F, (l,m)) 1x0e + 1x1o -> (A, (F,l,m)) Fx0e + Fx1o
        node_features = e3nn.flax.Linear(irreps)(node_features)
        node_features = node_features.mul_to_axis()  # (A, (F,l,m)) Fx0e + Fx1o -> (A, F, (l,m)) 1x0e + 1x1o
        s = node_features.filter('0e').array[..., 0]  # type: ignore
        v = node_features.filter('1o').array  # type: ignore
        return s, v

    @nn.compact
    def convert_to_irreps(self, s: FloatAxF, v: FloatAxFx3) -> e3nn.IrrepsArray:  # (A, F, (l,m))  with m,l as in Y_{l,m}
        s = e3nn.IrrepsArray('0e', s[..., None])  # type: ignore
        v = e3nn.IrrepsArray('1o', v)  # type: ignore
        out = e3nn.concatenate([s, v], axis=-1).axis_to_irreps()  # type: ignore
        out = e3nn.flax.Linear(
            self.atom_feature_dim * e3nn.Irreps(self.irreps_out)
        )(out)
        return out
