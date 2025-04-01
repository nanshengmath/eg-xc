import jax
import numpy as onp
import einops
import jax.numpy as jnp
from dataclasses import dataclass
from flax.struct import dataclass as flax_dataclass

from egxc.systems import Grid

from egxc.utils.typing import (
    Int1,
    Float1,
    FloatBxB,
    FloatN,
    FloatNxB,
    FloatSCF,
    FloatSCFxBxB,
    FloatAx3,
    FloatSCFxAx3,
    PRECISION,
)

from typing import Callable


@dataclass
class RelativeWeights:
    energy: float
    forces: float
    density: float

    def __post_init__(self):
        assert all(
            w >= 0 for w in self.__dict__.values()
        ), 'Weights must be non-negative.'


@dataclass
class LossConfig:
    """
    Configures the loss weights of the Kohn-Sham (KS) Regularizer
    See: Li et al. K. "Kohn-Sham Equations as Regularizer: Building Prior Knowledge into
    Machine-Learned Physics." Phys. Rev. Lett. 2021, 126 (3), 036401.
    https://doi.org/10.1103/PhysRevLett.126.036401.
    """

    weights: RelativeWeights
    decay_factors: FloatSCF
    reference_basis_is_same: bool = False

    def __post_init__(self):
        decay_factors = onp.array(self.decay_factors)
        self.decay_factors = decay_factors / decay_factors.sum()


def decay_only_final(cycles: int):
    out = onp.zeros(cycles)
    out[-1] = 1.0
    return out


def decay_dick2021(cycles: int, discard_first_n: int = 10) -> FloatSCF:
    """
    Weights as proposed by Dick et al. https://doi.org/10.1103/PhysRevB.104.L161109.
    """

    def w_j(j):
        return (
            j - discard_first_n
        ) ** 4  # NOTE: maybe there was a typo in the paper and this should be squared

    out = onp.fromfunction(w_j, (cycles,))
    out[:discard_first_n] = 0.0  # type: ignore
    return out


def decay_li2021(cycles: int, discard_first_n: int = 10) -> FloatSCF:
    """
    Weights from original publication on KS-Regularizer.
    """

    def w_j(j):
        return 0.9 ** (cycles - j) * (j > discard_first_n)

    return jnp.fromfunction(w_j, (cycles,))


def decay_egxc2025(cycles: int, discard_first_n: int):
    """
    Weights used in our paper: https://doi.org/10.48550/arXiv.2410.07972
    """
    consider_last_n = cycles - discard_first_n
    weights = jnp.arange(1, consider_last_n + 1) ** 2
    return jnp.hstack((jnp.zeros(discard_first_n), weights))


@flax_dataclass
class LossFns:
    energy: Callable[[Float1, FloatSCF], Float1]
    forces: Callable[[FloatAx3, FloatSCFxAx3], Float1]
    density: Callable[[FloatBxB, FloatSCFxBxB, Grid, Int1], Float1]


def get_loss_fns(config: LossConfig) -> LossFns:
    decay_factors = config.decay_factors

    def zero_fn(target, prediction):
        return jnp.array(0, dtype=PRECISION.loss)

    if config.weights.energy > 0.0:

        def energy_loss(target: Float1, prediction: FloatSCF) -> Float1:
            out = decay_factors * (target - prediction) ** 2
            return config.weights.energy * out.sum()
    else:
        energy_loss = zero_fn

    if config.weights.forces > 0.0:

        def forces_loss(target: FloatAx3, prediction: FloatSCFxAx3):
            out = jnp.einsum('c, af, caf->i', decay_factors, target, prediction)
            return config.weights.forces * out.sum()
    else:
        forces_loss = zero_fn

    if config.weights.density > 0.0:
        # raise NotImplementedError("Density Loss not yet implemented.")
        def density_fn(P, aos):
            return jnp.einsum('uv,iu,iv->i', P, aos, aos)

        if config.reference_basis_is_same:
            # exploit linearity of density function to avoid redundant computation
            def density_loss(  # type: ignore
                target: FloatBxB, prediction: FloatSCFxBxB, grid: Grid, n_electrons: Int1
            ) -> Float1:
                density_difference = jax.vmap(density_fn, in_axes=(0, None))(
                    target[None] - prediction, grid.aos
                )
                out = einops.einsum(
                    decay_factors, density_difference**2, grid.weights, 'scf, scf n, n ->'
                )
                return out / (n_electrons**2)
        else:
            # TODO: think about optimal projections between density matrices of different datasets
            raise NotImplementedError(
                'Density Loss for non matching basis sets not yet implemented.'
            )

            def density_loss(
                target: FloatSCF,
                prediction: FloatSCF,
                quad_weights: FloatN,
                aos: FloatNxB,
            ):
                # TODO: compute density on grid for both reference and predicted basis
                pass
    else:

        def density_loss(*args):
            return jnp.array(0, dtype=PRECISION.loss)

    return LossFns(energy_loss, forces_loss, density_loss)
