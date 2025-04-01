import jax
import jax.numpy as jnp
import flax.linen as nn

from egxc.xc_energy.functionals.base import (
    BaseEnergyFunctional,
    e_x_uniform_electron_gas,
)
from egxc.xc_energy.features import ueg_spin_pol_e_x_factor, transform_tau_to_alpha
from egxc.xc_energy.functionals.classical.lsda import pw92_correlation_energy_density

from .nn.base import FeatureMLP
from egxc.utils.typing import FloatN
from typing import Tuple, Callable


def I_transform(x: jax.Array, a: float) -> jax.Array:
    ex = jnp.exp(x)
    return a / (1 + (a - 1) * ex) - 1


def _input_transform(
    n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN
) -> Tuple[FloatN, FloatN, FloatN, FloatN]:
    n_t = n ** (1 / 3)
    zeta_t = ueg_spin_pol_e_x_factor(zeta)
    s_t = s
    tau_t = transform_tau_to_alpha(n, zeta, s, tau)

    eps_log = 1e-5

    n_t = jnp.log(n_t + eps_log)
    zeta_t = jnp.log(zeta_t + eps_log)
    s_t = -jnp.expm1(-(s_t**2)) * jnp.log1p(s_t)
    tau_t = jnp.log1p(tau_t) - jnp.log(2)
    return n_t, zeta_t, s_t, tau_t


class Dick2021(BaseEnergyFunctional):
    """
    Sebastian Dick and Marivi Fernandez-Serra.
    “Highly Accurate and Constrained Density Functional
    Obtained with Differentiable Programming.”
    Physical Review B 104, no. 16 (October 12, 2021): L161109.
    https://doi.org/10.1103/PhysRevB.104.L161109.
    """

    # Default used in their publication:
    n_layers: int = 4  # including the final output layer
    hidden_dim: int = 16
    activation: Callable[[jax.Array], jax.Array] = nn.gelu
    is_hybrid = False
    is_graph_based = False

    def setup(self) -> None:
        # TODO: check if they apply activation to last layer
        self.x_net = FeatureMLP(self.n_layers, self.hidden_dim, self.activation)
        self.c_net = FeatureMLP(self.n_layers, self.hidden_dim, self.activation)

    def xc_energy_density(  # type: ignore
        self, n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN
    ) -> FloatN:
        n_t, zeta_t, s_t, tau_t = _input_transform(n, zeta, s, tau)
        ueg_limit_factor = s_t + jnp.tanh(tau_t) ** 2
        NNx = self.x_net(s_t, tau_t)
        Fx = 1 + I_transform(NNx[:, 0] * ueg_limit_factor, 1.147)
        e_x = e_x_uniform_electron_gas(n) * Fx

        NNc = self.c_net(n_t, zeta_t, s_t, tau_t)
        Fc = 1 + I_transform(NNc[:, 0] * ueg_limit_factor, 2)
        e_c = pw92_correlation_energy_density(n, zeta) * Fc
        return e_x + e_c
