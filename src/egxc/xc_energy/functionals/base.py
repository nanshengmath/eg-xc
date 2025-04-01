import jax
import jax.numpy as jnp
import flax.linen as nn

from egxc.xc_energy.features import DensityFeatures
from egxc.systems import Grid
from egxc.utils.typing import (
    BoolB,
    Float1,
    FloatN,
    PRECISION,
    FloatBxB,
    Float2xBxB,
)
from typing import Tuple, Any


def e_x_uniform_electron_gas(n: FloatN) -> FloatN:
    """
    The exchange energy per particle of the uniform electron gas.
    """
    return -(3 / 4) * (3 / jnp.pi) ** (1 / 3) * n ** (1 / 3)


def wiegner_seitz_radius(n: FloatN, epsilon: float = 0) -> FloatN:
    """
    The Wigner-Seitz radius
    """
    return (3 / (4 * jnp.pi * n + epsilon)) ** (1 / 3)


def fermi_wave_vector(n: FloatN) -> FloatN:
    """
    The Fermi wave vector commonly denoted as k_f
    """
    return (3 * jnp.pi**2 * n) ** (1 / 3)


class BaseEnergyFunctional(nn.Module):
    """
    A baseclass which includes numeric quadrature utilities for (semi-)local
    energy densities.
    """

    def __call__(
        self, weights: FloatN, *feats: FloatN, **non_local_kwargs: Any
    ) -> Float1:
        xc_energy = self.integrate_energy_density(weights, *feats)
        assert xc_energy.dtype == PRECISION.xc_energy
        return xc_energy

    def integrate_energy_density(self, weights: FloatN, *feats: FloatN) -> Float1:
        """
        weights (FloatN): Quadrature weights
        feats (FloatN): local electron density features in the following order:
            n (FloatN): electron density
            xi (FloatN): spin polarization
            s (FloatN): |grad n|
            tau (FloatN): reduced density gradient

        Returns: Float1: exchange-correlation energy
        """
        n = feats[0]
        e_xc = self.xc_energy_density(*feats)
        return (weights * n * e_xc).sum()

    def xc_energy_density(self, *args: FloatN) -> FloatN:
        """
        Abstract method for the exchange-correlation energy density.
        """
        raise NotImplementedError


class XCModule(nn.Module):
    xc_functional: BaseEnergyFunctional
    feature_fn: DensityFeatures

    def __call__(
        self,
        density_matrix: FloatBxB | Float2xBxB,
        grid: Grid,
        **non_local_kwargs: Any,
    ) -> Float1:
        return self.xc_energy(density_matrix, grid, **non_local_kwargs)

    def xc_energy(
        self,
        density_matrix: FloatBxB | Float2xBxB,
        grid: Grid,
        **non_local_kwargs: Any,
    ) -> Float1:
        mask, feats = self.feature_fn(density_matrix, grid.aos, grid.grad_aos)
        if self.xc_functional.is_hybrid:
            non_local_kwargs['density_matrix'] = density_matrix
        return self.xc_functional(grid.weights * mask, *feats, **non_local_kwargs)

    def xc_potential(
        self,
        density_matrix: FloatBxB | Float2xBxB,
        grid: Grid,
        basis_mask: BoolB,
        **non_local_kwargs: Any,
    ) -> FloatBxB:
        V = jax.grad(self.xc_energy, argnums=0)(density_matrix, grid, **non_local_kwargs)
        return jnp.where(basis_mask[:, None] * basis_mask[None, :], V, 0.0)  # type: ignore

    def xc_energy_and_potential(
        self,
        density_matrix: FloatBxB | Float2xBxB,
        grid: Grid,
        basis_mask: BoolB,
        **non_local_kwargs: Any,
    ) -> Tuple[Float1, FloatBxB]:
        e, V = jax.value_and_grad(self.xc_energy, argnums=0)(density_matrix, grid, **non_local_kwargs)
        return e, jnp.where(basis_mask[:, None] * basis_mask[None, :], V, 0.0)  # type: ignore
