import pytest
import jax.numpy as jnp
from jax import random
import jax
import optax

from egxc.solver import fock, scf
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.learnable import Nagai2020, Dick2021
from egxc.systems import examples, System
from egxc.systems.base import nuclear_energy

from egxc.utils.typing import ElectRepTensorType as ERTT
from utils import set_jax_testing_config


set_jax_testing_config()


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
@pytest.mark.parametrize(
    'functional',
    [Nagai2020(hidden_dim=8), Dick2021(hidden_dim=8)],
    ids=['Nagai2020', 'Dick2021'],
)
def test_overfit_h2o(spin_restricted, functional):
    ert_type = ERTT.DENSITY_FITTED
    xc_mod = fock.XCModule(functional, DensityFeatures(spin_restricted))
    scf_solver = scf.SelfConsistentFieldSolver(
        xc_mod, 15, ert_type, spin_restricted, 'DIIS'
    )
    psys = examples.get_preloaded(
        'water', '6-31G(d)', ert_type=ert_type, spin_restricted=spin_restricted, alignment=1
    )

    sys = System.from_preloaded(psys)

    params = scf_solver.init(random.PRNGKey(0), psys.initial_density_matrix, sys)

    def loss_fn(params):
        TARGET_ENERGY = -76.38566321214728  # RKS B3LYP/6-31G(d) from PySCF
        energies, _ = scf_solver.apply(params, psys.initial_density_matrix, sys)
        e_final = (energies[0] + energies[1])[-1] + nuclear_energy(sys._nuc_pos, sys)
        loss = (e_final - TARGET_ENERGY) ** 2
        return loss, e_final

    opt = optax.adam(1e-3)  # deliberately small learning rate, to guarantee loss decrease
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, energy

    loss_list = []
    for i in range(10):
        params, opt_state, loss, energy = step(params, opt_state)
        print(f'Iteration {i}, loss: {loss}, energy: {energy}')
        loss_list.append(loss)

    loss = jnp.array(loss_list)

    assert jnp.all(loss[1:] < loss[:-1]), 'Loss is not decreasing'


