import jax
import jax.numpy as jnp
from functools import partial

from typing import (
    Generic,
    NamedTuple,
    TypeVar,
)


T = TypeVar('T')


class EMA(NamedTuple, Generic[T]):
    data: T
    weight: float

    @classmethod
    def create(cls, tree: T) -> 'EMA[T]':
        return cls(jax.tree_map(lambda x: jnp.zeros_like(x), tree), 0)


@partial(jax.jit, static_argnames=('decay',))
def update(ema: EMA[T], value: T, decay: float) -> EMA[T]:
    return EMA(
        jax.tree.map(lambda a, b: a * decay + b, ema.data, value),
        ema.weight * decay + 1
    )


@jax.jit
def value(ema: EMA[T]) -> T:
    return jax.tree.map(
        lambda x: x / ema.weight, ema.data
    )
