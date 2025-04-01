import e3nn_jax as e3nn

from egxc.xc_energy.functionals.learnable.nn import Encoder, PaiNN, Decoder

from egxc.xc_energy.features import DensityFeatures
from egxc.systems import examples

from utils import call_module_as_function, set_jax_testing_config, PyscfSystemWrapper

set_jax_testing_config()


def test_l1():
    irreps = e3nn.Irreps('0e + 1o')
    cutoff = 5.0
    encoder = Encoder(irreps, cutoff)
    density_feat_fn = DensityFeatures(spin_restricted=True)

    sys = examples.get('water', 'sto-3g', alignment=0)
    P = PyscfSystemWrapper(sys, basis='sto-3g').density_matrix
    _, (n, _) = call_module_as_function(density_feat_fn, P, sys.grid.aos, None)
    args = (sys._nuc_pos, sys.atom_mask, sys.grid.coords, sys.grid.weights, n)

    node_feats, cache = call_module_as_function(encoder, *args)
    assert node_feats.shape == (3, 33, 4)

    FEATURES = 128
    gnn = PaiNN(FEATURES, cutoff, 1)
    _, node_feats = call_module_as_function(gnn, node_feats, sys._nuc_pos, sys.atom_mask)

    assert node_feats.shape == (3, FEATURES, 4)  # type: ignore

    decoder = Decoder(FEATURES)
    spatial_feats = call_module_as_function(decoder, node_feats, cache)

    assert spatial_feats.shape == (len(sys.grid.weights), FEATURES)  # type: ignore