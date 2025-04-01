import flax.linen as nn

from egxc.systems.base import System
from egxc.utils.typing import (
    FloatBxB,
    Float2xBxB,
    FloatSCF,
    FloatSCFxBxB,
    FloatSCFx2xBxB,
)

from typing import Tuple


class Solver(nn.Module):
    """
    Abstract base class for all solvers.
    """

    def __call__(
        self,
        initial_density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> Tuple[Tuple[FloatSCF, FloatSCF], FloatSCFxBxB | FloatSCFx2xBxB]: ...
