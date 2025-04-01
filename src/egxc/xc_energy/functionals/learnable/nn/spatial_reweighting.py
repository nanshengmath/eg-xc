import jax
import jax.numpy as jnp
import flax.linen as nn
from egxc.xc_energy.functionals.learnable.nn import base

from typing import Callable


class SpatialReweighting(nn.Module):
    """
    A spatial reweighting network that learns the importance of different
    """

    layers: int
    hidden_dim: int
    activation: Callable[[jax.Array], jax.Array] = nn.silu

    def setup(self):
        self.reweighting_net = base.FeatureMLP(
            self.layers, self.hidden_dim, self.activation, concatenate=True
        )
        self.reweighting_scale = self.param(
            'reweighting_scale', jax.nn.initializers.zeros, (), jnp.float32
        )
        self.local_base_factor = self.param(
            'local_factor', jax.nn.initializers.ones, (), jnp.float32
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.local_base_factor + self.reweighting_scale * self.reweighting_net(x)
        return out.squeeze(axis=-1)
