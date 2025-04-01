"""
Adapted from H. Helal's and A Fitzgibbon's
"MESS: Modern Electronic Structure Simulations" package
https://arxiv.org/abs/2406.03121
"""

from enum import Enum, unique, auto

import jax.numpy as jnp
from flax.struct import dataclass

from egxc.xc_energy.functionals.classical import lsda
from egxc.xc_energy.functionals.base import (
    e_x_uniform_electron_gas,
    fermi_wave_vector,
    BaseEnergyFunctional,
)
from egxc.xc_energy.features import transform_s_to_abs_grad_n

from egxc.utils.typing import FloatN


def e_x_b88(n: FloatN, s: FloatN) -> FloatN:
    beta = 0.0042 * 2 ** (1 / 3)
    abs_grad_n = transform_s_to_abs_grad_n(n, s)
    x = abs_grad_n / n ** (4 / 3)
    d = 1 + 6 * beta * x * jnp.arcsinh(2 ** (1 / 3) * x)
    e_x = e_x_uniform_electron_gas(n) - beta * n ** (1 / 3) * x**2 / d
    return e_x


def e_x_pbe(n: FloatN, s: FloatN) -> FloatN:
    beta = 0.066725  # Eq 4
    mu = beta * jnp.pi**2 / 3  # Eq 12
    kappa = 0.8040  # Eq 14

    F = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
    return e_x_uniform_electron_gas(n) * F


def e_c_pbe(n: FloatN, zeta: FloatN, s: FloatN) -> FloatN:
    beta = 0.066725
    gamma = (1 - jnp.log(2.0)) / jnp.pi**2
    phi = 0.5 * (jnp.power(1 + zeta, 2 / 3) + jnp.power(1 - zeta, 2 / 3))
    ec_pw = lsda.pw92_correlation_energy_density(n, zeta)
    A = beta / gamma * (jnp.exp(-ec_pw / (gamma * phi**3)) - 1) ** -1  # Eq 8
    kf = fermi_wave_vector(n)
    ks = jnp.sqrt(4 * kf / jnp.pi)
    abs_grad_n = transform_s_to_abs_grad_n(n, s)
    t = abs_grad_n / (2 * phi * ks * n)
    u = 1 + beta / gamma * t**2 * (1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)
    H = gamma * phi**3 * jnp.log(u)  # Eq 7
    return ec_pw + H


def e_c_lyp(
    n: FloatN,
    s: FloatN,
) -> FloatN:
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    CF = 0.3 * (3 * jnp.pi**2) ** (2 / 3)

    x_n = n ** (-1 / 3)

    v = 1 + d * x_n
    omega = jnp.exp(-c * x_n) / v * n ** (-11 / 3)
    delta = c * x_n + d * x_n / v
    abs_grad_n = transform_s_to_abs_grad_n(n, s)
    g = (1 / 24 + 7 * delta / 72) * n * abs_grad_n ** 2

    e_c = -a / v - a * b * omega * (CF * n ** (11 / 3) - g)
    return e_c


@unique
class GGAExchangeType(Enum):
    PBE = auto()
    B88 = auto()


@unique
class GGACorrelationType(Enum):
    PBE = auto()
    LYP = auto()


@dataclass
class GGAType:
    x: GGAExchangeType
    c: GGACorrelationType


class GGA(BaseEnergyFunctional):
    """
    Local density approximation.
    """

    xc_type: GGAType = GGAType(GGAExchangeType.PBE, GGACorrelationType.PBE)
    is_hybrid = False
    is_graph_based = False

    def setup(self):
        print(
            f'Using GGA with {self.xc_type.x.name} exchange and {self.xc_type.c.name} correlation.'
        )

    def xc_energy_density(  # type: ignore
        self, n: FloatN, zeta: FloatN, s: FloatN
    ) -> FloatN:
        if self.xc_type.x == GGAExchangeType.PBE:
            e_x = e_x_pbe(n, s)
        elif self.xc_type.x == GGAExchangeType.B88:
            e_x = e_x_b88(n, s)
        else:
            raise ValueError('Invalid exchange type.')

        if self.xc_type.c == GGACorrelationType.PBE:
            e_c = e_c_pbe(n, zeta, s)
        elif self.xc_type.c == GGACorrelationType.LYP:
            e_c = e_c_lyp(n, s)
        else:
            raise ValueError('Invalid correlation type.')
        return e_x + e_c
