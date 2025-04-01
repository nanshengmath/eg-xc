import numpy as onp

import jax
import jax.numpy as jnp

from numpy.typing import ArrayLike

FloatN = onp.ndarray
FloatNx3 = onp.ndarray
FloatBxB = onp.ndarray
Float2xBxB = onp.ndarray
FloatBxBxBxB = onp.ndarray


def relative_error(value: ArrayLike, target: ArrayLike, eps=1e-15) -> jax.Array:
    return jnp.abs(value - target) / (jnp.abs(target) + eps)  # type: ignore


def is_close(value: jax.Array, target: jax.Array, tolerance: float, absolute=False):
    """
    Returns True if the error is less than the tolerance.
    if absolute is True, the error is the absolute difference between value and target,
    otherwise it is the relative error.
    """
    if absolute:
        error = jnp.abs(value - target)
    else:
        error = relative_error(value, target)
    return jnp.all(error < tolerance)


def assert_is_close(
    value: jax.Array,
    target: jax.Array,
    mask: jax.Array | None = None,
    tolerance: float = 1e-10,
    absolute=False,
    name='',
):
    """
    Asserts that two scalars or arrays are close element-wise.
    If mask is not None, only compare the elements where mask is True.
    "tolerance" is the maximum relative error allowed if absolute is False,
    otherwise it is the maximum absolute error allowed.
    """
    assert (
        value.shape == target.shape
    ), f'Shapes do not match: {value.shape} != {target.shape}'
    if mask is not None:
        value = value[mask]
        target = target[mask]
    rel = relative_error(value, target)
    abs = jnp.abs(value - target)
    if rel.size != 1:  # check if scalar
        rel = rel.flatten()
        abs = abs.flatten()
        if absolute:
            idx = abs.argmax()
        else:
            idx = rel.argmax()
        rel = rel[idx]
        abs = abs[idx]
    else:
        assert mask is None, 'Mask cannot be used with scalar values'
    if absolute:
        assert abs < tolerance, f'{name}: max (rel) / abs error: ({rel:.2e}) / {abs:.2e}'
    else:
        assert rel < tolerance, f'{name}: max rel / (abs) error: {rel:.2e} / ({abs:.2e})'

