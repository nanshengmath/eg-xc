import jax
import jax.numpy as jnp
from functools import partial

from egxc.xc_energy.functionals.base import XCModule
from egxc.solver import fock
from egxc.solver.scf.diis import DiisState, diis_update
from egxc.solver import linalg
from egxc.solver.base import Solver

from egxc.systems.base import System
from egxc.utils.typing import (
    FloatBxB,
    Float2xBxB,
    FloatSCF,
    FloatSCFxBxB,
    FloatSCFx2xBxB,
    ElectRepTensorType,
)

from typing import Tuple, Literal

ConvAccState = DiisState | Tuple[FloatBxB | Float2xBxB, int] | None

ScfCycleCarry = Tuple[
    FloatBxB | Float2xBxB,
    FloatBxB | Float2xBxB,
    System,
    ConvAccState,
]


class SelfConsistentFieldSolver(Solver):
    XCModule: XCModule
    cycles: int
    ert_type: ElectRepTensorType
    spin_restricted: bool = True
    convergence_acceleration_method: Literal["Vanilla", "Momentum", "DIIS"] = "DIIS"

    def setup(self) -> None:
        self.FockModule = fock.FockMatrix(
            self.XCModule, self.ert_type, self.spin_restricted
        )  # type: ignore
        # set up specified convergence acceleration method to dampen oscillations of the fock matrix
        if self.convergence_acceleration_method == "DIIS":
            init_fn = partial(DiisState.init, self.cycles)
            if not self.spin_restricted:  # vmap over spin
                self.convergence_acc_fn = jax.vmap(diis_update, in_axes=(None, 0, 0, 0, None))
                init_fn = jax.vmap(init_fn, in_axes=(0, 0, None))
            else:
                self.convergence_acc_fn = diis_update
            self.init_convergence_acc_state = init_fn
        elif self.convergence_acceleration_method == "Momentum":
            self.init_convergence_acc_state = lambda F, *args: (F, 0)
            def update_fn(cycle, F_raw, state, *args):
                F_previous = state
                alpha = 0.3**cycle + 0.3  # FIXME: make this a hyperparameter
                F = alpha * F_raw + (1 - alpha) * F_previous
                return F, F
            self.convergence_acc_fn = update_fn
        elif self.convergence_acceleration_method == "Vanilla":
            # Default to vanilla SCF
            self.init_convergence_acc_state = lambda *args: None
            self.convergence_acc_fn = lambda _, F, *args: (F, None)
        else:
            raise ValueError(
                f"Invalid convergence acceleration method: {self.convergence_acceleration_method}"
            )

        def new_density_matrix(F, X, occupancies):
            _, C = linalg.modified_generalized_eigenvalue_problem(F, X)
            return linalg.coeff_to_density_matrix(C, occupancies)

        if not self.spin_restricted:  # vmap over spin
            new_density_matrix = jax.vmap(new_density_matrix, in_axes=(0, None, 0))
        self.new_density_matrix = new_density_matrix

    def __call__(  # TODO: think about whether nuc gradient should stop here?
        self,
        initial_density_matrix: FloatBxB | Float2xBxB,
        sys: System,
    ) -> Tuple[Tuple[FloatSCF, FloatSCF], FloatSCFxBxB | FloatSCFx2xBxB]:
        initial_fock_matrix = self.FockModule.fock_matrix(
            sys._nuc_pos, initial_density_matrix, sys
        )
        energies, density_matrices = self.scf_loop(
            initial_fock_matrix,
            initial_density_matrix,
            sys
        )
        return energies, density_matrices

    def scf_loop(
        self, F_0, P_0, sys
    ) -> Tuple[Tuple[FloatSCF, FloatSCF], FloatSCFxBxB | FloatSCFx2xBxB]:
        """
        DIIS loop for SCF convergence.
        Args:
            F_0: Initial Fock matrix
            P_0: Initial density matrix
            cst: Constant system tensors
            sys: System
        Returns:
            Energies: Array of energies for each cycle (total_cycles)
            Density matrices: Array of density matrices for each cycle (total_cycles, N_bas, N_bas)
        """

        def loop_body(
            carry: ScfCycleCarry, cycle: int
        ) -> Tuple[ScfCycleCarry, FloatBxB | Float2xBxB]:
            F, P, sys, acc_state = carry
            P = self.new_density_matrix(F, sys.fock_tensors.diagonal_overlap, sys.fock_tensors.occupancies)
            F = self.FockModule.fock_matrix(sys._nuc_pos, P, sys)
            F, acc_state = self.convergence_acc_fn(cycle, F, acc_state, P, sys.fock_tensors)  # type: ignore
            return (F, P, sys, acc_state), P

        acc_state = self.init_convergence_acc_state(F_0, P_0, sys.fock_tensors)
        init_state = (F_0, P_0, sys, acc_state)

        _, density_matrices = jax.lax.scan(
            loop_body, init_state, xs=jnp.arange(self.cycles)  # type: ignore
        )
        energies = self.__calc_energies_along_scf_trajectory(
            sys._nuc_pos, density_matrices, sys
        )
        return energies, density_matrices

    def __calc_energies_along_scf_trajectory(self, nuc_pos, density_matrices, sys):
        energy_fn = jax.vmap(self.FockModule.energy, in_axes=(None, 0, None))
        return energy_fn(nuc_pos, density_matrices, sys)
