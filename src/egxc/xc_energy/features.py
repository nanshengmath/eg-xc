import jax.numpy as jnp
import flax.linen as nn

from egxc.utils.typing import (
    BoolN,
    FloatBxB,
    Float2xBxB,
    FloatN,
    FloatNxB,
    FloatNxBx3,
)
from typing import Tuple


def transform_abs_grad_n_to_s(n: FloatN, abs_grad_n: FloatN) -> FloatN:
    """
    Computes the reduced density gradient s which can be expressed using
    the fermi wave vector k_f as: s = |grad n| / (2 * k_f * n)

    TODO: check in previous implementation we needed to do:
    s = np.clip(s, 0) is this still the case?
    """
    return abs_grad_n / (2 * (3 * jnp.pi**2) ** (1 / 3) * n ** (4 / 3))


def transform_s_to_abs_grad_n(n: FloatN, s: FloatN) -> FloatN:
    """
    Computes the absolute gradient of the electron density |grad n|
    """
    return 2 * (3 * jnp.pi**2) ** (1 / 3) * n ** (4 / 3) * s


def ueg_spin_pol_e_x_factor(zeta: FloatN) -> FloatN:
    """
    Computes the dependency factor of the exchange energy of the uniformly
    spin-polarized electron gas on the spin polarization zeta.

    Oliver, G. L.; Perdew, J. P.
    "Spin-Density Gradient Expansion for the Kinetic Energy."
    Phys. Rev. A 1979, 20 (2), 397–403.
    https://doi.org/10.1103/PhysRevA.20.397.
    """
    return (1 / 2) * ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3))


def ueg_spin_pol_e_kin_factor(zeta: FloatN) -> FloatN:
    """
    Computes the dependency factor of the kinetic energy of the uniformly
    spin-polarized electron gas on the spin polarization zeta and the kinetic
    energy density tau.
    """
    return (1 / 2) * ((1 + zeta) ** (5 / 3) + (1 - zeta) ** (5 / 3))


def transform_tau_to_alpha(n: FloatN, zeta: FloatN, s: FloatN, tau: FloatN) -> FloatN:
    """
    Computes the kinetic energy density parameter alpha
    """
    tau_w = (1 / 2) * (3 * jnp.pi**2) ** (2 / 3) * n ** (5 / 3) * s**2
    d_zeta = ueg_spin_pol_e_kin_factor(zeta)
    tau_unif = (3 / 10) * (3 * jnp.pi**2) ** (2 / 3) * n ** (5 / 3) * d_zeta
    alpha = (tau - tau_w) / tau_unif
    return alpha


def _mask_density(threshold: float, n: FloatN) -> Tuple[BoolN, FloatN]:
    """
    avoid divide by zero when n = 0 by replacing with threshold
    """
    mask = n > threshold
    n = jnp.where(mask, n, threshold)
    return mask, n


class DensityFeatures(nn.Module):
    spin_restricted: bool
    min_density_threshold: float = 1e-15

    def __call__(
        self,
        density_matrix: FloatBxB | Float2xBxB,
        aos: FloatNxB,
        grad_aos: FloatNxBx3 | None,
    ) -> Tuple[BoolN, Tuple[FloatN, FloatN]] | Tuple[BoolN, Tuple[FloatN, FloatN, FloatN, FloatN]]:
        """
        Computes the electron density features:
            n >= 0(density),
            zeta in [-1,1] (spin-polarization)
            s >= 0(reduced density gradient)
            tau >= 0(kinetic energy density)
        if grad_aos is None:
            returns mask, (n, zeta)
        else:
            returns mask, (n, zeta, s, tau)
        where mask is a boolean array to mask out densities below the threshold
        to avoid divisions by zero.
        """
        if self.spin_restricted:
            assert density_matrix.ndim == 2
            return self._spin_restricted_feats(density_matrix, aos, grad_aos)
        else:
            assert density_matrix.ndim == 3
            return self._spin_unrestricted_feats(density_matrix, aos, grad_aos)

    def _spin_restricted_feats(
        self, density_matrix: FloatBxB, aos: FloatNxB, grad_aos: FloatNxBx3 | None
    ) -> Tuple[BoolN, Tuple[FloatN, FloatN]] | Tuple[BoolN, Tuple[FloatN, FloatN, FloatN, FloatN]]:
        n = jnp.einsum('uv,iu,iv->i', density_matrix, aos, aos)
        mask, n = _mask_density(self.min_density_threshold, n)
        zeta = jnp.zeros_like(n)
        if grad_aos is None:
            return mask, (n, zeta)
        _n_grad = jnp.einsum(
            'uv,iuj,iv -> ij', density_matrix, grad_aos, aos
        ) + jnp.einsum('uv,iu,ivj -> ij', density_matrix, aos, grad_aos)
        abs_n_grad = jnp.linalg.norm(_n_grad, axis=-1)
        s = transform_abs_grad_n_to_s(n, abs_n_grad)
        tau = 0.5 * jnp.einsum('uv,iuj,ivj->i', density_matrix, grad_aos, grad_aos)
        return mask, (n, zeta, s, tau)

    def _spin_unrestricted_feats(
        self, density_matrix: Float2xBxB, aos: FloatNxB, grad_aos: FloatNxBx3 | None
    ) -> Tuple[BoolN, Tuple[FloatN, FloatN]] | Tuple[BoolN, Tuple[FloatN, FloatN, FloatN, FloatN]]:
        n_up = jnp.einsum('uv,iu,iv->i', density_matrix[0], aos, aos)
        n_down = jnp.einsum('uv,iu,iv->i', density_matrix[1], aos, aos)
        n = n_up + n_down
        mask, n = _mask_density(self.min_density_threshold, n)
        zeta = (n_up - n_down) / n  # TODO: check for division by zero
        if grad_aos is None:
            return mask, (n, zeta)
        _n_grad = jnp.einsum(
            'suv,iuj,iv -> ij', density_matrix, grad_aos, aos
        ) + jnp.einsum('suv,iu,ivj -> ij', density_matrix, aos, grad_aos)
        abs_n_grad = jnp.linalg.norm(_n_grad, axis=-1)
        s = transform_abs_grad_n_to_s(n, abs_n_grad)
        tau = 0.5 * jnp.einsum('suv,iuj,ivj->i', density_matrix, grad_aos, grad_aos)
        return mask, (n, zeta, s, tau)
