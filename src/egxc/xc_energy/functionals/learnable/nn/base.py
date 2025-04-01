import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn
import jaxtyping

from typing import Sequence, Callable, Tuple
from egxc.utils.typing import FloatAxNxRBF

IntT = jaxtyping.Int[jaxtyping.Array, 'T']
EmbeddingCache = Tuple[int, IntT, e3nn.IrrepsArray, FloatAxNxRBF]


def shifted_softplus(x: jax.Array) -> jax.Array:
    return jnp.logaddexp2(x, 0) - 1


class MLP(nn.Module):
    dims: Sequence[int]
    activation: Callable[[jax.Array], jax.Array]
    init_last_layer_to_zero: bool = False
    apply_activation_to_output: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for dim in self.dims[:-1]:
            x = nn.Dense(dim)(x)
            x = self.activation(x)

        last_layer_kernel_init = (
            nn.initializers.zeros
            if self.init_last_layer_to_zero
            else nn.initializers.lecun_normal()
        )

        x = nn.Dense(self.dims[-1], kernel_init=last_layer_kernel_init)(x)
        if self.apply_activation_to_output:
            x = self.activation(x)
        return x


class FeatureMLP(nn.Module):
    """
    A simple multi-layer perceptron to act on (semi)local electron density features.
    """

    n_layers: int
    hidden_dim: int
    activation: Callable[[jax.Array], jax.Array]
    output_dim: int = 1
    init_last_layer_to_zero: bool = False
    apply_activation_to_output: bool = False
    concatenate: bool = False

    def setup(self):
        dims = [self.hidden_dim] * (self.n_layers - 1) + [self.output_dim]
        self.mlp = MLP(
            dims,
            self.activation,
            self.init_last_layer_to_zero,
            self.apply_activation_to_output,
        )

    def __call__(self, *feats: jax.Array) -> jax.Array:
        """
        Concatenate the input features if more than one and pass them through the MLP.
        """
        if self.concatenate:
            x = jnp.concatenate(feats, axis=-1)
        else:
            x = jnp.stack(feats, axis=-1)
        return self.mlp(x)
