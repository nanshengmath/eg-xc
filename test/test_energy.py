from jax import random
import pytest

from egxc.solver import fock
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.classical import mgga
from egxc.systems import examples
from egxc.systems.base import nuclear_energy

from utils import PyscfSystemWrapper as PySys
from utils import assert_is_close, set_jax_testing_config

from egxc.utils.typing import ElectRepTensorType as ERTT

set_jax_testing_config()


def test_nuclear_energy():
    basis = 'sto-3g'
    sys = examples.get('ethanol', basis)
    pyscf_sys = PySys(sys, basis)
    e_nuc = nuclear_energy(sys._nuc_pos, sys)
    e_ref = pyscf_sys.nuclear_repulsion_energy
    assert abs(e_nuc - e_ref) < 1e-6, f'Nuclear energy does not match {e_nuc} != {e_ref}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_fock_module(spin_restricted):
    basis = 'sto-3g'
    ert_type = ERTT.EXACT
    xc_mod = fock.XCModule(mgga.MetaGGA(), DensityFeatures(spin_restricted))
    fock_mod = fock.FockMatrix(xc_mod, ert_type, spin_restricted)
    sys = examples.get('h2o', basis=basis, alignment=1, ert_type=ert_type)
    pyscf_sys = PySys(
        sys,
        basis,
        xc='SCAN',
        grid_level=1,
        spin_restricted=spin_restricted,
    )

    H_ref = pyscf_sys.core_hamiltonian
    assert_is_close(sys.fock_tensors.core_hamiltonian, H_ref, name='core hamiltonian')  # type: ignore
    S_ref = pyscf_sys.overlap
    assert_is_close(sys.fock_tensors.overlap, S_ref, name='overlap matrix')  # type: ignore

    P = pyscf_sys.density_matrix
    nuc_pos = sys._nuc_pos

    fock_mod.init(random.PRNGKey(0), nuc_pos, P, sys)
    (e_hj, e_xc), F = fock_mod.apply(
        {}, nuc_pos, P, sys, method=fock_mod.energy_and_fock_matrix
    )

    F_ref = pyscf_sys.fock_matrix
    mask = F_ref > 1e-15
    assert_is_close(F, F_ref, name='fock matrix', mask=mask, tolerance=1e-7)  # type: ignore

    e_ref_xc = pyscf_sys.xc_energy
    assert abs(e_xc - e_ref_xc) < 1e-6, f'XC energy does not match {e_xc} != {e_ref_xc}'

    e_tot = e_xc + e_hj + nuclear_energy(nuc_pos, sys)
    e_ref = pyscf_sys.total_energy
    assert abs(e_tot - e_ref) < 1e-6, f'Total energy does not match {e_tot} != {e_ref}'
    if not spin_restricted:
        assert P.shape == F.shape  # type: ignore
        assert F.ndim == 3  # type: ignore
        assert F_ref.ndim == 3
