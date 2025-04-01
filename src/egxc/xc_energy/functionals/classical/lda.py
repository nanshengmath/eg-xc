from enum import Enum, unique, auto
import jax.numpy as jnp

from egxc.xc_energy.functionals.base import (
    e_x_uniform_electron_gas,
    wiegner_seitz_radius,
    BaseEnergyFunctional,
)
from egxc.xc_energy.functionals.classical import lsda
from egxc.utils.typing import FloatN


def pz81_correlation_energy_density(n: FloatN) -> FloatN:
    """
    Compute the LDA perdew-zunger correlation energy density.
    Perdew, J. P.; Zunger, A. Self-Interaction Correction to Density-Functional
    Approximations for Many-Electron Systems. Phys. Rev. B 1981, 23 (10)
    https://doi.org/10.1103/PhysRevB.23.5048.

    Args:
        n (jax.Array): electron density
    """
    # Constants
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    rs = wiegner_seitz_radius(n)
    rs_sqrt = jnp.sqrt(rs)
    e_c = jnp.where(
        rs >= 1,
        gamma / (1 + beta1 * rs_sqrt + beta2 * rs),
        A * jnp.log(rs) + B + C * rs * jnp.log(rs) + D * rs,
    )
    return e_c

@unique
class LDACorrelationType(Enum):
    VWN5 = auto()
    PZ81 = auto()
    PW92_spin_restricted = auto()


class LDA(BaseEnergyFunctional):
    """
    Local density approximation.
    """
    correlation_type: LDACorrelationType = LDACorrelationType.VWN5
    is_hybrid = False
    is_graph_based = False

    def setup(self):
        print(f"Using LDA with {self.correlation_type.name} correlation.")

    def xc_energy_density(self, n: FloatN, _) -> FloatN:  # type: ignore
        e_x = e_x_uniform_electron_gas(n)
        if self.correlation_type == LDACorrelationType.VWN5:
            zeta = jnp.zeros_like(n)
            e_c = lsda.vwn5_correlation_energy_density(n, zeta)
        elif self.correlation_type == LDACorrelationType.PZ81:
            e_c = pz81_correlation_energy_density(n)
        elif self.correlation_type == LDACorrelationType.PW92_spin_restricted:
            zeta = jnp.zeros_like(n)
            e_c = lsda.pw92_correlation_energy_density(n, zeta)
        else:
            raise ValueError('Invalid correlation type.')
        return e_x + e_c
