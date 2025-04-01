from enum import Enum, unique, auto
import jax
import jax.numpy as jnp

from egxc.xc_energy.functionals.base import (
    e_x_uniform_electron_gas,
    wiegner_seitz_radius,
    BaseEnergyFunctional,
)

from egxc.utils.typing import FloatN


def vwn5_correlation_energy_density(n: FloatN, zeta: FloatN, use_rpa=False) -> FloatN:
    """
    Compute the VWN5 correlation energy density.
    Vosko, S. H.; Wilk, L.; Nusair, M. Accurate Spin-Dependent Electron Liquid
    Correlation Energies for Local Spin Density Calculations: A Critical Analysis.
    Can. J. Phys. 1980, 58 (8), 1200–1211. https://doi.org/10.1139/p80-159.

    Args:
        n (FloatN): electron density
        zeta (FloatN): spin polarization
        use_rpa (bool): whether to use the random phase approximation

    implementation taken from H. Helal's and A Fitzgibbon's
    "MESS: Modern Electronic Structure Simulations" package
    https://arxiv.org/abs/2406.03121
    """

    # https://gitlab.com/libxc/libxc/-/blob/devel/maple/vwn.mpl?ref_type=heads
    # paramagnetic (eps_0) / ferromagnetic (eps_1) / spin stiffness (alpha)
    A = jnp.array([0.0310907, 0.5 * 0.0310907, -1 / (6 * jnp.pi**2)])

    if use_rpa:
        x0 = jnp.array([-0.409286, -0.743294, -0.228344])
        b = jnp.array([13.0720, 20.1231, 1.06835])
        c = jnp.array([42.7198, 101.578, 11.4813])
    else:
        # https://math.nist.gov/DFTdata/atomdata/node5.html
        x0 = jnp.array([-0.10498, -0.32500, -4.75840e-3])
        b = jnp.array([3.72744, 7.06042, 1.13107])
        c = jnp.array([12.9352, 18.0578, 13.0045])

    def f(xi):
        u = (1 + xi) ** (4 / 3) + (1 - xi) ** (4 / 3) - 2
        v = 2 * (2 ** (1 / 3) - 1)
        return u / v

    def d2fdz20():
        grad_f = jax.grad(jax.grad(f))
        return float(grad_f(0.0))

    F2 = d2fdz20()

    rs = jnp.power(3 / (4 * jnp.pi * n), 1 / 3).reshape(-1, 1)
    x = jnp.sqrt(rs).reshape(-1, 1)
    X = rs + b * x + c
    X0 = x0**2 + b * x0 + c
    Q = jnp.sqrt(4 * c - b**2)

    u = jnp.log(x**2 / X) + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
    v = jnp.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * jnp.arctan(Q / (2 * x + b))
    ec = A * (u - b * x0 / X0 * v)
    e0, e1, alpha = ec.T
    beta = F2 * (e1 - e0) / alpha - 1
    eps_c = e0 + alpha * f(zeta) / F2 * (1 + beta * zeta**4)
    return eps_c


def pw92_correlation_energy_density(
    n: FloatN, zeta: FloatN, use_RPA=False, modified=False
) -> FloatN:
    """
    The correlation energy (LSDA1) of the uniform electron gas by
    Perdew and Wang (1992) (PW92).
    https://doi.org/10.1103/PhysRevB.45.13244,

    libxc reference implementation:
    https://github.com/ElectronicStructureLibrary/libxc/blob/master/src/lda_c_pw.c
    https://github.com/ElectronicStructureLibrary/libxc/blob/master/src/maple2c/lda_exc/lda_c_pw.c#L14
    https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/lda_exc/lda_c_pw.mpl
    https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/util.mpl
    """
    r_s = wiegner_seitz_radius(n)

    def analytic_base_from(p, A, a1, b1, b2, b3, b4) -> jax.Array:
        """
        p and A are constrained the remaining parameters were fitted (see ref)
        """
        beta_sum = (
            b1 * r_s ** (1 / 2) + b2 * r_s + b3 * r_s ** (3 / 2) + b4 * r_s ** (p + 1)
        )
        log_term = jnp.log1p(1 / (2 * A * beta_sum))
        return -2 * A * (1 + a1 * r_s) * log_term

    A_unpolarized = 0.031091 if not modified else 0.0310907
    A_polarized = 0.015545 if not modified else 0.01554535
    A_alpha_c = 0.016887 if not modified else 0.0168869
    if use_RPA:
        # random-phase approximation (RPA)
        ec_unpolarized = analytic_base_from(
            p=0.75,
            A=A_unpolarized,
            a1=0.082477,
            b1=5.1486,
            b2=1.6483,
            b3=0.2347,
            b4=0.20614,
        )
        # fmt: off
        ec_polarized = analytic_base_from(
            p=0.75, A=A_polarized, a1=0.035374, b1=6.4869, b2=1.3083, b3=0.15180, b4=0.082349,
        ) # fmt: on
        alpha_c = -analytic_base_from(
            p=1, A=A_alpha_c, a1=0.028829, b1=10.357, b2=3.6231, b3=0.47990, b4=0.12279
        )
    else:
        ec_unpolarized = analytic_base_from(
            p=1, A=A_unpolarized, a1=0.21370, b1=7.5957, b2=3.5876, b3=1.6382, b4=0.49294
        )
        ec_polarized = analytic_base_from(
            p=1, A=A_polarized, a1=0.20548, b1=14.1189, b2=6.1977, b3=3.3662, b4=0.62517
        )
        alpha_c = -analytic_base_from(
            p=1, A=A_alpha_c, a1=0.11125, b1=10.357, b2=3.6231, b3=0.88026, b4=0.49671
        )

    dd_f_zero = 1.709921 if not modified else 1.709920934161365617563962776245
    # alpha_c = dd_f_zero * (ec_polarized - ec_unpolarized)  # from another reference
    f_xi = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)

    return (
        ec_unpolarized
        + alpha_c * f_xi / dd_f_zero * (1 - zeta**4)
        + (ec_polarized - ec_unpolarized) * f_xi * zeta**4
    )


@unique
class LSDACorrelationType(Enum):
    PW92 = auto()


class LSDA(BaseEnergyFunctional):
    """
    Local spin density approximation.
    """
    correlation_type: LSDACorrelationType = LSDACorrelationType.PW92
    is_hybrid = False
    is_graph_based = False

    def setup(self) -> None:
        print(f"Using LSDA with {self.correlation_type.name} correlation.")

    def xc_energy_density(self, density: FloatN, zeta: FloatN) -> FloatN:  # type: ignore
        e_x = e_x_uniform_electron_gas(density)
        if self.correlation_type == LSDACorrelationType.PW92:
            e_c = pw92_correlation_energy_density(density, zeta)
        else:
            raise ValueError('Invalid correlation type.')
        return e_x + e_c

