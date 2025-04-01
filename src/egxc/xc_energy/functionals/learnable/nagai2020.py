import jax
import jax.numpy as jnp
import flax.linen as nn

from egxc.xc_energy.functionals.base import BaseEnergyFunctional
from egxc.xc_energy.features import ueg_spin_pol_e_x_factor, ueg_spin_pol_e_kin_factor

from .nn.base import FeatureMLP

from egxc.utils.typing import FloatN, FloatNx3
from typing import Callable


def weight(r1: FloatNx3, r2: FloatNx3, sigma: float) -> FloatN:
    """
    Computes the weight function used in the Nagai2020 exchange-correlation
    """
    return jnp.exp(-jnp.abs(r1 - r2) / sigma)


class Nagai2020(BaseEnergyFunctional):
    """
    Nagai, Ryo, Ryosuke Akashi, and Osamu Sugino.
    “Completing Density Functional Theory by Machine Learning Hidden Messages from Molecules.”
    Npj Computational Materials 6, no. 1 (May 5, 2020): 1–8.
    https://doi.org/10.1038/s41524-020-0310-0.

    https://github.com/ml-electron-project/NNfunctional

    TODO: add non-local features?
    """

    # Default used in their publication:
    n_layers: int = 4  # including the final output layer
    hidden_dim: int = 100
    activation: Callable[[jax.Array], jax.Array] = nn.elu
    epsilon: float = 1e-8
    is_hybrid = False
    is_graph_based = False

    def setup(self) -> None:
        self.net = FeatureMLP(
            self.n_layers,
            self.hidden_dim,
            self.activation,
            apply_activation_to_output=True,  # Follows the original implementation by Nagai et al.
            # See e.g. https://github.com/ml-electron-project/NNfunctional/blob/master/LSDA.py
            concatenate=True
        )

    def xc_energy_density(  # type: ignore
        self, n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN
    ) -> FloatN:
        # https://github.com/ml-electron-project/NNfunctional/blob/master/metaGGA.py
        n_t = n ** (1 / 3)
        s_t = s
        xi_t = ueg_spin_pol_e_x_factor(zeta)
        d_zeta = ueg_spin_pol_e_kin_factor(zeta)
        tau_t = tau / (n_t**5 * d_zeta)
        return -n_t * xi_t * self.learnable_factor(n_t, xi_t, s_t, tau_t)

    def learnable_factor(
        self, n_t: FloatN, xi_t: FloatN, s_t: FloatN, tau_t: FloatN
    ) -> FloatN:
        x = jnp.stack([n_t, xi_t, s_t, tau_t], axis=-1)
        x = jnp.log(x + self.epsilon)
        return 1 + self.net(x)[:, 0]

