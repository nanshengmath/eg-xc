import pytest
import jax.numpy as jnp
from jax import random
import jax
import optax

from egxc.solver import fock, scf
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.learnable import Dick2021
from egxc.systems import examples, System
from egxc.systems.base import nuclear_energy

from egxc.utils.typing import ElectRepTensorType as ERTT, Alignment
from utils import set_jax_testing_config, call_module_as_function


set_jax_testing_config()


def get_energy(alignment: int | Alignment, spin_restricted: bool):
    ert_type = ERTT.DENSITY_FITTED
    xc_mod = fock.XCModule(Dick2021(hidden_dim=8), DensityFeatures(spin_restricted))
    scf_solver = scf.SelfConsistentFieldSolver(
        xc_mod, 15, ert_type, spin_restricted, 'DIIS'
    )
    psys = examples.get_preloaded(
        'h2',
        'sto-3g',
        ert_type=ert_type,
        spin_restricted=spin_restricted,
        alignment=alignment,
    )
    sys = System.from_preloaded(psys)

    initial_xc_energy = call_module_as_function(
        xc_mod, psys.initial_density_matrix, sys.grid
    )
    assert initial_xc_energy.dtype == jnp.float64  # type: ignore

    energies, _ = call_module_as_function(scf_solver, psys.initial_density_matrix, sys)

    e_final = (energies[0] + energies[1])[-1] + nuclear_energy(sys._nuc_pos, sys)
    return e_final, initial_xc_energy


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_scf_cycle_grid_padding(spin_restricted):
    unpadded = get_energy(1, spin_restricted)

    padded = get_energy(Alignment(1, 1, 512), spin_restricted)  # no atom wise padding
    assert not jnp.isnan(padded[0])
    assert (
        padded[1] - unpadded[1] < 1e-12  # type: ignore
    ), f'Initial xc energies should not be affected by padding, got delta of {padded[1] - unpadded[1]}'  # type: ignore
    assert (
        padded[0] == unpadded[0]
    ), f'Final energies should not be affected by padding, got {padded[0]} and {unpadded[0]}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_scf_cycle_atom_padding(spin_restricted):
    unpadded = get_energy(1, spin_restricted)

    padded = get_energy(Alignment(12, 1, 1), spin_restricted)  # no atom wise padding
    assert not jnp.isnan(padded[0])
    assert (
        padded[1] - unpadded[1] < 1e-12  # type: ignore
    ), f'Initial xc energies should not be affected by padding, got delta of {padded[1] - unpadded[1]}'  # type: ignore
    assert (
        padded[0] == unpadded[0]
    ), f'Final energies should not be affected by padding, got {padded[0]} and {unpadded[0]}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_scf_cycle_basis_padding(spin_restricted):
    unpadded = get_energy(1, spin_restricted)

    padded = get_energy(Alignment(1, 3, 1), spin_restricted)  # no atom wise padding
    assert not jnp.isnan(padded[0])
    assert (
        padded[1] - unpadded[1] < 1e-12  # type: ignore
    ), f'Initial xc energies should not be affected by padding, got delta of {padded[1] - unpadded[1]}'  # type: ignore
    assert (
        padded[0] - unpadded[0] < 1e-12  # type: ignore
    ), f'Final energies should not be affected by padding, got {padded[0]} and {unpadded[0]}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_scf_cycle_full_padding(spin_restricted):
    unpadded = get_energy(1, spin_restricted)

    padded = get_energy(Alignment(4, 12, 512), spin_restricted)  # no atom wise padding
    assert not jnp.isnan(padded[0])
    assert (
        padded[1] - unpadded[1] < 1e-12  # type: ignore
    ), f'Initial xc energies should not be affected by padding, got delta of {padded[1] - unpadded[1]}'  # type: ignore
    assert (
        padded[0] - unpadded[0] < 1e-12  # type: ignore
    ), f'Final energies should not be affected by padding, got {padded[0]} and {unpadded[0]}'


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_update_step(spin_restricted):
    def update_step(alignment):
        ert_type = ERTT.DENSITY_FITTED
        xc_mod = fock.XCModule(Dick2021(hidden_dim=8), DensityFeatures(spin_restricted))
        scf_solver = scf.SelfConsistentFieldSolver(
            xc_mod, 15, ert_type, spin_restricted, 'DIIS'
        )
        psys = examples.get_preloaded(
            'h2o',
            'sto-3g',
            ert_type=ert_type,
            spin_restricted=spin_restricted,
            alignment=alignment,
        )

        sys = System.from_preloaded(psys)

        params = xc_mod.init(random.PRNGKey(0), psys.initial_density_matrix, sys.grid)
        assert (
            xc_mod.apply(params, psys.initial_density_matrix, sys.grid).dtype  # type: ignore
            == jnp.float64
        )
        assert (
            xc_mod.apply(
                params,
                psys.initial_density_matrix,
                sys.grid,
                sys.fock_tensors.basis_mask,
                method=xc_mod.xc_potential,
            ).dtype  # type: ignore
            == jnp.float64
        )

        params = scf_solver.init(random.PRNGKey(0), psys.initial_density_matrix, sys)

        def loss_fn(params):
            TARGET_ENERGY = -76.38566321214728  # RKS B3LYP/6-31G(d) from PySCF
            energies, _ = scf_solver.apply(params, psys.initial_density_matrix, sys)
            e_final = (energies[0] + energies[1])[-1] + nuclear_energy(sys._nuc_pos, sys)
            loss = (e_final - TARGET_ENERGY) ** 2
            return loss, e_final

        opt = optax.adam(
            1e-3
        )  # deliberately small learning rate, to guarantee loss decrease
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, energy

        loss_list = []
        for i in range(2):
            params, opt_state, loss, energy = step(params, opt_state)
            print(f'Iteration {i}, loss: {loss}, energy: {energy}')
            loss_list.append(loss)

        return loss_list

    loss1 = update_step(Alignment(1, 1, 1))

    loss2 = update_step(Alignment(1, 1, 512))
    assert loss2[1] < loss2[0], f'Loss is not decreasing with grid padding {loss2}'
    loss3 = update_step(Alignment(12, 1, 1))
    assert loss3[1] < loss3[0], f'Loss is not decreasing with atom padding {loss3}'
    loss4 = update_step(Alignment(1, 4, 1))
    assert loss4[1] < loss4[0], f'Loss is not decreasing with basis padding {loss4}'

    assert (
        loss2[0] + loss3[0] + loss4[0] - 3 * loss1[0] < 1e-12
    ), f'Loss deviates {loss1[0]} {loss2[0]} {loss3[0]} {loss4[0]}'
    assert (
        loss2[1] + loss3[1] + loss4[1] - 3 * loss1[1] < 1e-12
    ), f'Loss deviates {loss1[1]} {loss2[1]} {loss3[1]} {loss4[1]}'
