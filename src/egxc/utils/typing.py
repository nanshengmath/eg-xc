from types import SimpleNamespace

import jax.numpy as jnp
from jax import config
from jaxtyping import Array, Bool, Float, Int, PyTree

Float3 = Float[Array, '3']
FloatNx3 = Float[Array, 'N 3']
FloatN = Float[Array, 'N']
Float3xNxN = Float[Array, '3 N N']
FloatNxN = Float[Array, 'N N']
FloatNxM = Float[Array, 'N M']
Int1 = Int[Array, '1']
Int3 = Int[Array, '3']
IntN = Int[Array, 'N']
BoolN = Bool[Array, 'N']


NnParams = PyTree

__HIGH_PRECISION = jnp.float64 if config.jax_enable_x64 else jnp.float32
__DEFAULT_PRECISION = jnp.float32
__LOW_PRECISION = jnp.float32

precision = SimpleNamespace(
    _high=__HIGH_PRECISION,
    _default=__DEFAULT_PRECISION,
    _low=__LOW_PRECISION,
    basis=__HIGH_PRECISION,
)
