from scipy import linalg as ref_linalg
import numpy as onp
import jax.numpy as jnp
import pytest

from egxc.solver import fock, linalg, scf
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.classical import mgga
from egxc.systems import examples
from egxc.systems.base import nuclear_energy

from utils import PyscfSystemWrapper as PySys
from utils import call_module_as_function, set_jax_testing_config

from egxc.utils.typing import ElectRepTensorType as ERTT

set_jax_testing_config()


def test_generalized_eigenvalue_solver():
    onp.random.seed(0)
    n = 5
    F = onp.random.randint(-5, 5, size=(n, n))
    F = F + F.T  # symmetrize
    S = onp.random.randint(-2, 2, size=(n, n)).astype(onp.float64)
    S = S.T @ S  # symmetric positive definite
    X = linalg.transformation_matrix(jnp.array(S, dtype=jnp.float64))  # type: ignore
    e, C = linalg.modified_generalized_eigenvalue_problem(F, X)  # type: ignore
    e_ref, C_ref = ref_linalg.eigh(F, S)
    assert onp.allclose(e, e_ref), f'{e} != {e_ref}'
    # note that the sign of the eigenvectors is arbitrary
    assert onp.allclose(onp.abs(C), onp.abs(C_ref)), f'{C} != {C_ref}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
@pytest.mark.parametrize('conv_acc_method', ['Vanilla', 'DIIS'], ids=['Vanilla', 'DIIS'])
def test_scf_method(spin_restricted, conv_acc_method):
    basis = '6-31G(d)'
    ert_type = ERTT.EXACT
    CYCLES = 25
    xc_mod = fock.XCModule(mgga.MetaGGA(), DensityFeatures(spin_restricted))

    scf_solver = scf.SelfConsistentFieldSolver(
        xc_mod, CYCLES, ert_type, spin_restricted, conv_acc_method
    )

    sys = examples.get(
        'ethanol', basis, ert_type=ert_type, alignment=1, spin_restricted=spin_restricted
    )
    pyscf_sys = PySys(
        sys,
        basis,
        xc='SCAN',
        grid_level=1,
        spin_restricted=spin_restricted,
    )

    if not conv_acc_method == 'DIIS':
        pyscf_sys.mf.diis = None
    pyscf_sys.mf.max_cycle = CYCLES
    pyscf_sys.mf.kernel()

    P_0 = pyscf_sys.initial_density_matrix

    (e_hj, e_xc), density_matrices = call_module_as_function(
        scf_solver, P_0, sys, jit=True
    )  # type: ignore

    e_tot = e_xc + e_hj + nuclear_energy(sys._nuc_pos, sys)
    e_tot = e_tot[-1]  # energy of last cycle
    e_ref = pyscf_sys.total_energy
    assert (
        abs(e_tot - e_ref) < 3e-6
    ), f'Total energy does not match {e_tot:.8e} != {e_ref:.8e}, difference {(e_tot - e_ref):.3e} Ha'  # type: ignore
