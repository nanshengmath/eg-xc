"""
This module contains the implementation of the SCAN (Strongly Constrained and
Appropriately Normed Semilocal) meta-GGA functional by Sun et al.
https://doi.org/10.1103/PhysRevLett.115.036402
https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.115.036402

The SCAN functional is a meta-GGA functional, which means that it depends on:
    r_s = (3 / (4 * pi * n))**(1 / 3)
the spin polarization:
    xi  = (n_up - n_down) / n
the reduced gradient:
    s = |grad(n)| / (2 * (3 * pi**2)**(1 / 3) * n**(4 / 3))
"""

from enum import Enum, unique, auto

import jax
from functools import partial
import jax.numpy as jnp

from egxc.xc_energy.functionals.base import (
    e_x_uniform_electron_gas,
    wiegner_seitz_radius,
    BaseEnergyFunctional,
)
from egxc.xc_energy.functionals.classical.lsda import pw92_correlation_energy_density
from egxc.xc_energy.features import transform_tau_to_alpha
from egxc.utils.typing import FloatN


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _f_interp(alpha: FloatN, c1: float, c2: float, d: float) -> FloatN:
    term1 = jnp.exp(-c1 * alpha / (1 - alpha))
    term2 = -d * jnp.exp(c2 / (1 - alpha))
    return jnp.where(alpha < 1, term1, term2)


@_f_interp.defjvp
def jvp_f_interp(c1, c2, d, primals, tangents):
    """
    does not account for derivative w.r.t. constants c1, c2, d
    """
    alpha, = primals
    alpha_dot, = tangents
    df = _f_interp(alpha, c1, c2, d)
    dterm1_factor = -c1 / (1 - alpha) ** 2
    dterm2_factor = c2 / (1 - alpha) ** 2
    df_dot = jnp.where(alpha < 1, dterm1_factor, dterm2_factor) * df * alpha_dot
    return df, df_dot


@jax.jit
def e_x_scan(n: FloatN, s: FloatN, alpha: FloatN) -> FloatN:
    """
    The exchange energy per particle of SCAN
    """

    def F_x(s: FloatN, alpha: FloatN) -> FloatN:
        """
        The exchange enhancement factor
        """
        # fit parameters
        k1 = 0.065
        c1x = 0.667
        c2x = 0.8
        dx = 1.24

        def h1x_fn(s: jax.Array, alpha: jax.Array) -> jax.Array:
            mu_ak = 10 / 81
            b2 = jnp.sqrt(5913 / 405000)
            b1 = (511 / 13500) / (2 * b2)
            b3 = 0.5
            b4 = mu_ak**2 / k1 - 1606 / 18225 - b1**2

            exp1 = jnp.exp(-jnp.abs(b4) * s**2 / mu_ak)
            exp2 = jnp.exp(-b3 * (1 - alpha) ** 2)
            x = (
                mu_ak * s**2 * (1 + (b4 * s**2 / mu_ak) * exp1)
                + (b1 * s**2 + b2 * (1 - alpha) * exp2) ** 2
            )
            return 1 + k1 - k1 / (1 + x / k1)

        a1 = 4.9479
        h0x = 1.174

        def gx(s: jax.Array) -> jax.Array:
            return -jnp.expm1(-a1 / jnp.sqrt(s))

        h1x = h1x_fn(s, alpha)
        fx_alpha = _f_interp(alpha, c1x, c2x, dx)
        return (h1x + fx_alpha * (h0x - h1x)) * gx(s)

    return e_x_uniform_electron_gas(n) * F_x(s, alpha)


@jax.jit
def e_c_scan(n: FloatN, s: FloatN, zeta: FloatN, alpha: FloatN) -> FloatN:
    """
    The correlation energy per particle of SCAN
    TODO: verify zeta != 0 correctness if polarized systems are added

    https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/mgga_exc/mgga_c_scan.mpl
    """
    # fit parameters
    c1c = 0.64
    c2c = 1.5
    dc = 0.7
    # fixed constants
    b1c = 0.0285764
    b2c = 0.0889
    b3c = 0.125541

    def Psi_fn(zeta: jax.Array) -> jax.Array:
        return ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2

    def ec1_fn(n: jax.Array, s: jax.Array, zeta: jax.Array) -> jax.Array:
        """
        Perdew-Ernzerhof-Wang 1996 (PEW96)-like correlation energy
        https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/gga_exc/gga_c_pbe.mpl
        """
        ec_LSDA1 = pw92_correlation_energy_density(n, zeta)

        def H1(r_s: jax.Array, s: jax.Array, zeta: jax.Array) -> jax.Array:
            # gamma = 0.031091
            gamma = (1 - jnp.log(2)) / jnp.pi**2
            # beta = 0.06672455060314922
            beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)  # SCAN
            # beta = 0.066725  # PBE0
            Psi = Psi_fn(zeta)
            w1 = jnp.expm1(-ec_LSDA1 / (gamma * Psi**3))
            A = beta / (gamma * w1)
            t = ((3 * jnp.pi**2 / 16) ** (1 / 3) * s) / (Psi * jnp.sqrt(r_s))
            g = (1 + 4 * A * t**2) ** (-1 / 4)
            # g = 1 / (1 + A * t**2 + A**2 * t**4)  # PBE0
            return gamma * Psi**3 * jnp.log1p(w1 * (1 - g))

        return ec_LSDA1 + H1(r_s, s, zeta)

    def ec0_fn(r_s: jax.Array, s: jax.Array, zeta: jax.Array) -> jax.Array:
        ec_LDA0 = -b1c / (1 + b2c * jnp.sqrt(r_s) + b3c * r_s)
        dx = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3)) / 2
        Gc = (1 - 2.3631 * (dx - 1)) * (1 - zeta**12)
        w0 = jnp.expm1(-ec_LDA0 / b1c)
        chi_unpolarized = 0.12802585262625815  # 0.128026
        g = 1 / (1 + 4 * chi_unpolarized * s**2) ** (1 / 4)
        H0 = b1c * jnp.log1p(w0 * (1 - g))
        return (ec_LDA0 + H0) * Gc

    r_s = wiegner_seitz_radius(n)
    ec1 = ec1_fn(n, s, zeta)
    fc_alpha = _f_interp(alpha, c1c, c2c, dc)
    return ec1 + fc_alpha * (ec0_fn(r_s, s, zeta) - ec1)


@jax.jit
def e_xc_scan(n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN) -> FloatN:
    """
    The exchange-correlation energy per particle of SCAN
    """
    alpha = transform_tau_to_alpha(n, zeta, s, tau)
    return e_x_scan(n, s, alpha) + e_c_scan(n, s, zeta, alpha)


@unique
class MGGAType(Enum):
    SCAN = auto()


class MetaGGA(BaseEnergyFunctional):
    mgga_type: MGGAType = MGGAType.SCAN
    is_hybrid = False
    is_graph_based = False

    def setup(self) -> None:
        print(f"Using {self.mgga_type.name} meta-GGA functional.")

    def xc_energy_density(  # type: ignore
        self, n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN
    ) -> FloatN:
        if self.mgga_type == MGGAType.SCAN:
            e_xc = e_xc_scan(n, zeta, s, tau)
        else:
            raise ValueError('Invalid correlation type.')
        return e_xc
