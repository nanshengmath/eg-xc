import pytest

import jax
import jax.numpy as jnp
from egxc.solver import fock
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.classical import mgga
from egxc.systems import examples, System

from utils import PyscfSystemWrapper as PySys
from utils import assert_is_close, set_jax_testing_config

set_jax_testing_config()
jax.config.update('jax_debug_nans', True)

BASIS = '6-31G(d)'


@pytest.mark.parametrize(
    'align', [1, 4], ids=['without_padding', 'with_padding']
)
def test_xc_potential(align: int):
    spin_restricted = True
    xc_module = fock.XCModule(
        mgga.MetaGGA(),
        DensityFeatures(spin_restricted),
    )
    psys = examples.get_preloaded(
        'h2', BASIS, spin_restricted=spin_restricted, alignment=align
    )
    sys = System.from_preloaded(psys)
    P = psys.initial_density_matrix

    assert xc_module.xc_energy(P, sys.grid) != jnp.nan  # type: ignore
    V_xc = xc_module.xc_potential(P, sys.grid, sys.fock_tensors.basis_mask)  # type: ignore

    pyscf_sys = PySys(
        sys,
        BASIS,
        xc='SCAN',  # TODO try lda
        grid_level=1,
        spin_restricted=spin_restricted,
    )
    P = P[sys.fock_tensors.basis_mask, :][:, sys.fock_tensors.basis_mask]
    V_xc_ref = pyscf_sys.xc_potential(P)
    V_xc = V_xc[sys.fock_tensors.basis_mask, :][:, sys.fock_tensors.basis_mask]
    # TODO: should the indices belonging to padded basis functions be zero of V_xc?
    assert_is_close(
        V_xc,
        V_xc_ref,  # type: ignore
        name='exchange correlation potential',
        tolerance=1e-6,
        absolute=True,
    )  # type: ignore
