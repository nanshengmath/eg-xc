from functools import partial
import jax
import jax.numpy as jnp
import grain.python as grain

from egxc.systems import System, Grid
from egxc.systems.preload import PreloadSystem, preload_system_using_pyscf
from egxc.discretization import QuadratureGridFn, BasisFn
from egxc.dataloading.base import RawSample, Targets
from typing import Tuple, Callable, Sequence
from egxc.utils.typing import Alignment, ElectRepTensorType, FloatBxB


class PreloadTransform(grain.MapTransform):
    def __init__(
        self,
        basis: str,
        spin_restricted: bool,
        alignment: Alignment,
        ert_type: ElectRepTensorType | None,
        include_fock_tensors: bool,
        include_grid: bool,
        grid_level: int,
        center: bool,
    ):
        self.basis = basis
        self.spin_restricted = spin_restricted
        self.alignment = alignment
        self.ert_type = ert_type
        self.include_fock_tensors = include_fock_tensors
        self.include_grid = include_grid
        self.grid_level = grid_level
        self.center = center

    def map(self, raw_sample: RawSample) -> Tuple[PreloadSystem, Targets]:  # type: ignore
        (nuc_pos, atom_z, charge, spin), targets = raw_sample
        psys = preload_system_using_pyscf(
            nuc_pos,
            atom_z,
            charge=charge,
            spin=spin,
            basis=self.basis,
            spin_restricted=self.spin_restricted,
            alignment=self.alignment,
            center=self.center,
            include_fock_tensors=self.include_fock_tensors,
            ert_type=self.ert_type,
            include_grid=self.include_grid,
            grid_level=self.grid_level,
        )
        return psys, targets


def get_preload_transform(
    batch_size: int,
    basis: str,
    spin_restricted: bool,
    alignment: Alignment,
    ert_type: ElectRepTensorType,
    include_fock_tensors: bool,
    include_grid: bool = False,
    grid_level: int = 1,
    center: bool = False,
) -> Sequence[grain.Transformation]:
    preload_transform = PreloadTransform(
        basis=basis,
        spin_restricted=spin_restricted,
        alignment=alignment,
        ert_type=ert_type,
        include_fock_tensors=include_fock_tensors,
        include_grid=include_grid,
        grid_level=grid_level,
        center=center,
    )

    if batch_size > 1:
        transformations = (
            preload_transform,
            grain.Batch(
                batch_size=batch_size,
                drop_remainder=False,
            ),
        )
    else:
        transformations = (preload_transform,)

    return transformations


ToJaxTransform = Callable[[PreloadSystem], Tuple[FloatBxB, System]]


def get_jax_transform(
    grid_and_basis_fn: Tuple[QuadratureGridFn, BasisFn] | None,
    fock_tensors_fn: Callable | None,
) -> ToJaxTransform:
    def compute_grid(psys: PreloadSystem) -> Grid | None:
        if grid_and_basis_fn is None:
            return None
        else:
            grid_fn, basis_fn = grid_and_basis_fn
            coords, weights = grid_fn(
                psys.nuc_pos,
                psys.atom_z,  # type: ignore
                psys.atom_mask,  # type: ignore
            )
            aos = basis_fn(
                coords,
                psys.nuc_pos,  # type: ignore
                psys.atom_z.array,  # type: ignore
                psys.atom_mask,  # type: ignore
                psys.periods,  # type: ignore
                psys.max_number_of_basis_fns,
            )
            if isinstance(aos, tuple):
                aos, grad_aos = aos
                return Grid.create(coords, weights, aos, grad_aos)
            else:
                return Grid.create(coords, weights, aos, None)

    if fock_tensors_fn is not None:
        # TODO: implement
        raise NotImplementedError

    # @partial(jax.jit, donate_argnums=(0,))  TODO:
    def input_transform(psys: PreloadSystem) -> Tuple[FloatBxB, System]:
        grid = compute_grid(psys)
        sys = System.from_preloaded(psys, grid=grid)
        return jnp.asarray(psys.initial_density_matrix), sys

    return input_transform
