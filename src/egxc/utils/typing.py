import numpy as onp
import hashlib
from dataclasses import dataclass, field
from enum import Enum, unique, auto
from types import SimpleNamespace

from typing import Tuple, Set, Annotated
from numpy.typing import NDArray
from jaxtyping import Array, Bool, Float, Int, PyTree

### TYPING LEGEND: ###
# N: grid points
# A: atoms
# Z: elements in the periodic table
# B: basis functions
# E: electrons
# G: Gaussians for a basis function
# C_SPH: cartesian spherical harmonics
# M_SPH: real spherical harmonics
# Q: density fitting auxiliary basis
# SCF: number of scf iterations
# F: node features (atom features)

# dataloading
NpIntA = Annotated[NDArray[onp.int8], 'shape=(?,)']
NpFloatAx3 = Annotated[NDArray[onp.float64], 'shape=(?, 3)']
NpFloatBxB = Annotated[NDArray[onp.float64], 'shape=(?, ?)']

# Compile time static types
CompileStaticStr = str
CompileStaticInt = int
CompileStaticIntA = Tuple[int, ...]

# General
Int1 = Int[Array, '1']
Int3 = Int[Array, '3']
Float1 = Float[Array, '1']
Float3 = Float[Array, '3']

# Structure related
BoolA = Bool[Array, 'A']
BoolN = Bool[Array, 'N']
IntA = Int[Array, 'A']
FloatA = Float[Array, 'A']
FloatAx3 = Float[Array, 'A 3']
FloatSCFxAx3 = Float[Array, 'SCF A 3']
FloatAxA = Float[Array, 'A A']
FloatAxAx3 = Float[Array, 'A A 3']
FloatAxN = Float[Array, 'A N']
FloatAxNxRBF = Float[Array, 'A N RBF']
FloatAxG = Float[Array, 'A G']
IntE = Int[Array, 'E']
Bool2xE = Bool[Array, '2 E']
FloatE = Float[Array, 'E']
IntB = Int[Array, 'B']
BoolB = Bool[Array, 'B']
Bool2xB = Bool[Array, '2 B']
FloatB = Float[Array, 'B']
FloatBxE = Float[Array, 'B E']
Float2xBxE = Float[Array, '2 B E']
FloatBxB = Float[Array, 'B B']
Float2xBxB = Float[Array, '2 B B']
FloatZ = Float[Array, 'Z']
FloatZxG = Float[Array, 'Z G']
IntN = Int[Array, 'N']
FloatN = Float[Array, 'N']
FloatNx3 = Float[Array, 'N 3']
FloatNx4 = Float[Array, 'N 4']
FloatNxA = Float[Array, 'N A']
FloatNxAx3 = Float[Array, 'N A 3']
FloatNxF = Float[Array, 'N F']
FloatNxB = Float[Array, 'N B']
FloatNxBx3 = Float[Array, 'N B 3']
Float4xNxB = Float[Array, '4 N B']
BoolQ = Bool[Array, 'Q']
FloatQxBxB = Float[Array, 'Q B B']
FloatBxBxBxB = Float[Array, 'B B B B']
FloatSCF = Float[Array, 'SCF']
FloatSCFxSCF = Float[Array, 'SCF SCF']
FloatSCFxBxB = Float[Array, 'SCF B B']
FloatSCFx2xBxB = Float[Array, 'SCF 2 B B']

# GNN related
FloatAxF = Float[Array, 'A F']
FloatAxFx3 = Float[Array, 'A F 3']
FloatAxAx3F = Float[Array, 'A A 3F']

# Basis related
FloatG = Float[Array, 'G']
FloatM_SPH = Float[Array, 'M_SPH']
FloatNxM_SPH = Float[Array, 'N M_SPH']
FloatNxC_SPH = Float[Array, 'N C_SPH']
FloatAxNxM_SPH = Float[Array, 'N M_SPH']


NnParams = PyTree

__HIGH_PRECISION = "float64"
__LOW_PRECISION = "float32"

PRECISION = SimpleNamespace(
    basis=__HIGH_PRECISION,
    forces=__HIGH_PRECISION,
    xc_energy=__HIGH_PRECISION,
    quadrature=__HIGH_PRECISION,
    solver=__HIGH_PRECISION,
    loss=__HIGH_PRECISION,
)


@dataclass(frozen=True)
class HashableArray:
    array: NDArray = field(compare=False)

    def __post_init__(self):
        # Ensure the array is immutable
        object.__setattr__(self, 'array', onp.asarray(self.array))
        self.array.setflags(write=False)

    def __hash__(self) -> int:
        return hash(self._array_hash())

    def _array_hash(self) -> bytes:
        """Returns a hashable representation of the array."""
        return hashlib.sha256(self.array.tobytes()).digest()

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override NumPy's universal function handling to ensure `__eq__` is used."""
        if ufunc == np.equal and method == '__call__':
            return self.array.__eq__(inputs[1])
        return NotImplemented  # Let NumPy handle other operations

    def toset(self) -> Set:
        return set(self.array.tolist())


@dataclass(frozen=True)
class PermutiationInvariantHashableArray(HashableArray):
    def _array_hash(self) -> bytes:
        """Returns a hashable representation of the array."""
        return hashlib.sha256(onp.sort(self.array, axis=0).tobytes()).digest()



@unique
class ElectRepTensorType(Enum):
    """
    Enumeration for the type of electronic repulsion tensor.
    """

    EXACT = auto()
    DENSITY_FITTED = auto()


@dataclass
class Alignment:
    atom: int | Tuple[int, ...] = 1
    basis: int = 1
    grid: int = 1

    @property
    def is_aligned(self) -> bool:
        if isinstance(self.atom, int):
            is_atom_aligned = self.atom > 1
        else:
            is_atom_aligned = any(a > 1 for a in self.atom)
        return is_atom_aligned or self.basis > 1 or self.grid > 1