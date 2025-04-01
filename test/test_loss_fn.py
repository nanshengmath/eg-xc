import jax.numpy as jnp
import einops

from egxc.training import loss
from egxc.systems.examples import get

from utils import set_jax_testing_config, PyscfSystemWrapper

set_jax_testing_config()



def test_density_loss():
    n_cycles = 10
    loss_config = loss.LossConfig(
        loss.RelativeWeights(energy=0, forces=0, density=1.0),
        jnp.ones(n_cycles),
        reference_basis_is_same=True
    )
    loss_fns = loss.get_loss_fns(loss_config)

    sys = get('water', basis='6-31G(d)', alignment=1)
    test = PyscfSystemWrapper(sys, basis='6-31G(d)')
    P0 = test.initial_density_matrix
    P1 = test.density_matrix

    P1_predicted = einops.repeat(P1, 'i j -> scf i j', scf=n_cycles)

    # Test density loss
    density_loss = loss_fns.density(
        P1, P1_predicted, sys.grid, sys.n_electrons  # type: ignore
    )
    assert density_loss == 0.0

    density_loss = loss_fns.density(
        P0, P1_predicted, sys.grid, sys.n_electrons  # type: ignore
    )
    assert density_loss > 0.0