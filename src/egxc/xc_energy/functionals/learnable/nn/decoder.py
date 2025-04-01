import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn

from .base import MLP, EmbeddingCache
from egxc.utils.typing import FloatNxF
from typing import Callable


class Decoder(nn.Module):
    atom_feature_dim: int
    activation: Callable[[jax.Array], jax.Array] = nn.silu

    @nn.compact
    def __call__(
        self,
        atom_features: e3nn.IrrepsArray,
        cache: EmbeddingCache,
    ) -> FloatNxF:
        N, truncated_idx, spherical_harmonics, radial_basis_vals = cache

        _, _, RBF = radial_basis_vals.shape
        F = self.atom_feature_dim


        atom_features = atom_features.axis_to_mul()
        inv_features = atom_features.filter('0e').array  # type: ignore

        spatial_features = jnp.zeros((N, F))
        for sph_h, node_feats_h in zip(spherical_harmonics.chunks, atom_features.chunks):
            sph_h = sph_h.squeeze(-2)  # type: ignore
            rbf_to_f = MLP(
                [
                    self.atom_feature_dim,
                    self.atom_feature_dim,
                    radial_basis_vals.shape[-1] * F,
                ],
                activation=self.activation,  # type: ignore
            )(inv_features).reshape(-1, RBF, F) / jnp.sqrt(RBF)

            sparse_decoded_spatial_feats = jnp.einsum(
                'ntd,nfd,ntr,nrf->tf',
                sph_h,
                node_feats_h,
                radial_basis_vals,
                rbf_to_f,
            )
            spatial_features = (
                spatial_features.at[truncated_idx].add(sparse_decoded_spatial_feats)
            )

        return spatial_features





# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import e3nn_jax as e3nn

# from .base import MLP, EmbeddingCache
# from egxc.utils.typing import FloatNxF
# from typing import Callable


# class Decoder(nn.Module):
#     atom_feature_dim: int
#     activation: Callable[[jax.Array], jax.Array] = nn.silu

#     @nn.compact
#     def __call__(
#         self,
#         node_features: e3nn.IrrepsArray,  # (A, F, (l,m))  with m,l as in Y_{l,m}
#         cache: EmbeddingCache,
#     ) -> FloatNxF:
#         N, truncated_idx, spherical_harmonics, radial_basis_vals = cache

#         _, _, RBF = radial_basis_vals.shape
#         F = self.atom_feature_dim

#         inv_features = node_features.filter('0e').axis_to_mul().array  # type: ignore

#         rbf_to_f = MLP(
#             [
#                 self.atom_feature_dim,
#                 self.atom_feature_dim,
#                 radial_basis_vals.shape[-1] * F,
#             ],
#             activation=self.activation,  # type: ignore
#         )(inv_features).reshape(-1, RBF, F) / jnp.sqrt(RBF)

#         sparse_spatial_feats = jnp.einsum(
#             'atr,atm,afm,arf->tf',  # 'atoms grid RBF, atoms grid M, atoms F M, atoms RBF F -> grid F'
#             radial_basis_vals,
#             spherical_harmonics.array,  # last dim is the m as in Y_{l,m}, where m = -l, -l+1, ..., l
#             node_features.array,  # last dim is m as in Y_{l,m}, where m = -l, -l+1, ..., l
#             rbf_to_f,
#             # optimize=[(0, 3), (0, 1), (0, 1)],  TODO: test whether jit compilation takes useful contraction order
#         )

#         spatial_feats = (
#             jnp.zeros((N, F)).at[truncated_idx].add(sparse_spatial_feats)
#         )
#         return spatial_feats
