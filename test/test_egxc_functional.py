import pytest
import jax.numpy as jnp
from jax import random
import jax
import optax
import e3nn_jax as e3nn

from egxc.solver import scf
from egxc.xc_energy import XCModule
from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.classical.mgga import MetaGGA
from egxc.xc_energy.functionals.learnable import Dick2021, EGXC
from egxc.xc_energy.functionals.learnable.nn.painn import PaiNN
from egxc.xc_energy.functionals.learnable.nn.encoder import Encoder
from egxc.xc_energy.functionals.learnable.nn.decoder import Decoder
from egxc.xc_energy.functionals.learnable.nn.spatial_reweighting import SpatialReweighting
from egxc.systems import examples
from egxc.systems.base import nuclear_energy

from utils import (
    call_module_as_function,
    assert_is_close,
    PyscfSystemWrapper,
    set_jax_testing_config,
)
from egxc.utils.typing import ElectRepTensorType as ERTT

set_jax_testing_config()


def __get_initial_values(mol_str: str, spin_restricted, ert_type, basis='6-31G(d)'):
    sys = examples.get(
        mol_str,
        basis=basis,
        ert_type=ert_type,
        spin_restricted=spin_restricted,
        alignment=1,  # FIXME: align throws error at the moment
    )
    py_sys = PyscfSystemWrapper(sys, basis, spin_restricted=spin_restricted, xc='SCAN')
    P0 = py_sys.initial_density_matrix
    P = py_sys.density_matrix
    approximate_e_xc = py_sys.xc_energy
    return P0, P, sys, approximate_e_xc


def test_embedding_integral_truncation():
    spin_restricted = False
    ert_type = ERTT.DENSITY_FITTED
    _, P, sys, _ = __get_initial_values('3bpa', spin_restricted, ert_type, basis='sto-6g')
    mask, feats = call_module_as_function(
        DensityFeatures(spin_restricted), P, sys.grid.aos, sys.grid.grad_aos
    )
    n = feats[0]  # type: ignore
    embedding = Encoder(
        e3nn.Irreps('0e + 1o'), 5.0, _quadrature_points_per_atom_scaling=100
    )
    embedding_truncated = Encoder(e3nn.Irreps('0e + 1o'), 5.0)
    f, _ = call_module_as_function(
        embedding,
        sys._nuc_pos,
        sys.atom_mask,
        sys.grid.coords,
        sys.grid.weights * mask,
        n,
    )
    f_truncated, _ = call_module_as_function(
        embedding_truncated,
        sys._nuc_pos,
        sys.atom_mask,
        sys.grid.coords,
        sys.grid.weights * mask,
        n,
    )
    s = f.filter('0e').mul_to_axis().array[..., 0]  # type: ignore
    s_t = f_truncated.filter('0e').mul_to_axis().array[..., 0]  # type: ignore
    assert_is_close(s_t, s, tolerance=1e-3)

    v = f.filter('1o').mul_to_axis().array
    v_t = f_truncated.filter('1o').mul_to_axis().array
    assert_is_close(v_t, v, tolerance=4e-3)


@pytest.mark.parametrize(
    'spin_restricted', [True, False], ids=['restricted', 'unrestricted']
)
def test_xc_energy_eval(spin_restricted):
    atom_features = 16
    xc_functional = EGXC(
        local_model=MetaGGA(),
        encoder=Encoder(e3nn.Irreps('0e + 1o'), 5.0),
        gnn=PaiNN(atom_features, 5.0, 1, 5),
        decoder=Decoder(atom_features),
        spatial_reweighting=SpatialReweighting(2, 8),
        use_graph_readout=True
    )
    xc_mod = XCModule(
        xc_functional,
        DensityFeatures(spin_restricted),
    )
    ert_type = ERTT.DENSITY_FITTED
    _, P, sys, e_xc_ref = __get_initial_values('water', spin_restricted, ert_type)

    non_local_kwargs = {}
    non_local_kwargs['atom_mask'] = sys.atom_mask
    non_local_kwargs['nuc_pos'] = sys._nuc_pos
    non_local_kwargs['grid_coords'] = sys.grid.coords

    params = xc_mod.init(random.PRNGKey(0), P, sys.grid, **non_local_kwargs)

    @jax.jit
    def energy_fn(params):
        return xc_mod.apply(params, P, sys.grid, **non_local_kwargs)

    e_xc = energy_fn(params)

    assert (
        abs(e_xc - e_xc_ref) < 1e-2
    ), f'Implausible energy of untrained network {e_xc}, (reference {e_xc_ref})'


@pytest.mark.parametrize(
    'functional',
    [
        EGXC(
            Dick2021(n_layers=3, hidden_dim=8),
            Encoder(e3nn.Irreps('0e + 1o'), 4.0),
            PaiNN(8, 4.0, 1, 5),
            Decoder(8),
            SpatialReweighting(2, 4),
            use_graph_readout=True
        ),
        EGXC(
            Dick2021(n_layers=3, hidden_dim=8),
            Encoder(e3nn.Irreps('0e + 1o'), 4.0, nuclei_partitioning='Exponential'),
            PaiNN(8, 4.0, 1, 5),
            Decoder(8),
            SpatialReweighting(2, 4),
            use_graph_readout=True
        ),
        EGXC(
            Dick2021(n_layers=3, hidden_dim=8),
            Encoder(e3nn.Irreps('0e + 1o'), 4.0, nuclei_partitioning='Gaussian'),
            PaiNN(8, 4.0, 1, 5),
            Decoder(8),
            SpatialReweighting(2, 4),
            use_graph_readout=True
        ),
    ],
    ids=[
        'EG-XC:PaiNN+Dick2021',
        'EG-XC:PaiNN+Dick2021:Exp',
        'EG-XC:PaiNN+Dick2021:Gauss',
    ],
)
def test_overfit_water(functional, spin_restricted=False):
    ert_type = ERTT.DENSITY_FITTED
    P0, _, sys, _ = __get_initial_values('water', spin_restricted, ert_type)
    xc_mod = XCModule(functional, DensityFeatures(spin_restricted))
    scf_solver = scf.SelfConsistentFieldSolver(
        xc_mod, 15, ert_type, spin_restricted, 'DIIS'
    )

    params = scf_solver.init(random.PRNGKey(0), P0, sys)

    def loss_fn(params):
        TARGET_ENERGY = -76.38566321214728  # RKS B3LYP/6-31G(d) from PySCF
        energies, _ = scf_solver.apply(params, P0, sys)
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
