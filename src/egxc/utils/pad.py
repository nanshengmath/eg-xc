import jax.numpy as jnp

from egxc.utils.typing import FloatN, FloatNx3
from typing import Tuple


def calc_padding_size(size: int, align: int) -> int:
    """
    Returns the padding size required to fulfill the alignment
    """
    remainder = size % align
    if remainder == 0:
        return 0
    return align - remainder


def create_one_dim_mask(
    in_length: int, align: int | None = None, add: int | None = None
) -> jnp.ndarray:
    """
    Create a 1D mask for padding where
        in_length: int: the length of the input
        align: int: the alignment size
        add: int: the padding size
    Note: either align or add can be set, but not both
    """
    if add is None:
        assert align is not None, "Either pad_to_align or padding_size must be set"
        add = calc_padding_size(in_length, align)
    else:
        assert align is None, "Either pad_to_align or padding_size must be set"
        assert add >= 0, "Padding size must be positive"
    new_length = in_length + add
    out = jnp.ones(in_length, dtype=bool)
    out = jnp.pad(out, (0, new_length - in_length))
    return out


def create_n_dim_mask(shape: Tuple[int], pad_to_align: Tuple[int]) -> jnp.ndarray:
    dim = len(shape)
    new_shape = [
        shape[i] + calc_padding_size(shape[i], pad_to_align[i]) for i in range(dim)
    ]
    out = jnp.ones(shape, dtype=bool)
    out = jnp.pad(out, [(0, new_shape[i] - shape[i]) for i in range(dim)])
    return out


def pad_quadrature_grid(
    pad_to_align: int, coords: FloatNx3, weights: FloatN
) -> Tuple[FloatNx3, FloatN]:
    """
    For quadrature no padding mask is required as the weights can serve a dual use
    """
    padding_size = calc_padding_size(coords.shape[0], pad_to_align)
    coords = jnp.pad(coords, ((0, padding_size), (0, 0)))
    weights = jnp.pad(weights, (0, padding_size))
    return coords, weights
