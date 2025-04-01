import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
from pyscf import dft
from functools import partial
import pytest

from egxc.xc_energy.features import DensityFeatures
from egxc.xc_energy.functionals.base import e_x_uniform_electron_gas, XCModule
from egxc.xc_energy.functionals.classical import lda, lsda, gga, mgga, hybrid

from egxc.systems import examples

from utils import PyscfSystemWrapper as PySys
from utils import assert_is_close, call_module_as_function, set_jax_testing_config

from egxc.systems import System
from egxc.systems.preload import z_to_periods
from egxc.discretization import get_gto_basis_fn
from egxc.utils.typing import ElectRepTensorType, FloatBxB


set_jax_testing_config()


BASIS = '6-31G(d)'


class FeatureFactory(nn.Module):
    spin_restricted: bool

    def setup(self):
        self.basis_fn = get_gto_basis_fn(BASIS, max_period=2, deriv=1)
        self.features = DensityFeatures(self.spin_restricted)

    def __call__(self, sys: System, density_matrix: FloatBxB):
        periods = z_to_periods(sys.atom_z)  # type: ignore
        n_basis = density_matrix.shape[-1]
        aos, grad_aos = self.basis_fn(
            sys.grid.coords, sys._nuc_pos, sys.atom_z, sys.atom_mask, periods, n_basis
        )  # type: ignore
        mask, feats = self.features(density_matrix, aos, grad_aos)  # type: ignore
        return sys.grid.weights, mask, feats


@pytest.mark.parametrize('align', [1, 4], ids=['without_padding', 'with_padding'])
def test_density_features(align: int, spin_restricted: bool = True):
    factory = FeatureFactory(spin_restricted=spin_restricted)
    psys = examples.get_preloaded(
        'h2', BASIS, alignment=align, spin_restricted=spin_restricted
    )
    sys = System.from_preloaded(psys)
    P = psys.initial_density_matrix

    weights, mask, feats = call_module_as_function(factory, sys, P)  # type: ignore
    n, zeta, s, tau = feats
    assert jnp.all(zeta == 0)

    pyscf_sys = PySys(sys, BASIS, grid_level=1, spin_restricted=spin_restricted)
    b_mask = sys.fock_tensors.basis_mask
    P = P[:, b_mask][b_mask, :]
    ref_n, _, ref_s, ref_tau = pyscf_sys.get_density_features(P, coords=sys.grid.coords, format='egxc')  # type: ignore

    n_pad = n.shape[0] - ref_n.shape[0]  #  type: ignore
    ref_n = onp.pad(ref_n, ((0, n_pad)))

    ref_n = onp.clip(ref_n, 1e-15, None)

    mask = mask & (weights > 0)

    assert_is_close(n, ref_n, mask, name='density')  # type: ignore
    assert_is_close(s, ref_s, mask, name='reduced gradient')  # type: ignore
    assert_is_close(tau, ref_tau, mask, name='kinetic energy density')  # type: ignore


ethanol = examples.get('ethanol', BASIS, alignment=1)
ethanol_pyscf_sys = PySys(
    ethanol,
    BASIS,
    grid_level=1,
    spin_restricted=False,
)


@pytest.fixture
def ethanol_features():
    factory = FeatureFactory(spin_restricted=False)
    weights, mask, feats = call_module_as_function(  # type: ignore
        factory, ethanol, ethanol_pyscf_sys.density_matrix
    )
    ref_feats = ethanol_pyscf_sys.get_density_features()
    yield weights, mask, feats, ref_feats


def __truncate_feats(feats, int):
    """
    n, dn_dx, dn_dy, dn_dz, tau
    """
    up_feats, down_feats = feats
    return up_feats[:int], down_feats[:int]


def __ref_feats_to_n(ref_feats):
    n_up, n_down = __truncate_feats(ref_feats, 1)
    return n_up + n_down


def assert_total_xc_energy(mask, e, n, e_target, n_target, weights, tolerance=1e-7):
    val = (mask * e * n * weights).sum()
    target = (mask * e_target * n_target * weights).sum()
    assert_is_close(val, target, tolerance=tolerance)


lda_cases = {
    'lda_exchange': (e_x_uniform_electron_gas, 'slater,'),
    'lda_correlation_pz': (lda.pz81_correlation_energy_density, ',lda_c_pz'),
    'lda_correlation_vwn5': (
        partial(lsda.vwn5_correlation_energy_density, zeta=0),  # type: ignore
        ',vwn5',
    ),
}


@pytest.mark.parametrize('xcfunc,libxcstr', lda_cases.values(), ids=lda_cases.keys())
def test_lda_energy_densities(ethanol_features, xcfunc, libxcstr):
    weights, mask, feats, ref_feats = ethanol_features
    n = feats[0]
    e = xcfunc(n)
    n_target = __ref_feats_to_n(ref_feats)
    e_target = dft.libxc.eval_xc(libxcstr, __truncate_feats(ref_feats, 1), spin=1)[0]
    assert_total_xc_energy(mask, e, n, e_target, n_target, weights, tolerance=1e-6)


lsda_cases = {
    'lsda_correlation_pw92': (lsda.pw92_correlation_energy_density, ',lda_c_pw'),
}


@pytest.mark.parametrize('xcfunc,libxcstr', lsda_cases.values(), ids=lsda_cases.keys())
def test_lsda_energy_densities(ethanol_features, xcfunc, libxcstr):
    weights, mask, feats, ref_feats = ethanol_features
    n, zeta, _, _ = feats
    e = xcfunc(n, zeta)
    n_target = __ref_feats_to_n(ref_feats)
    e_target = dft.libxc.eval_xc(libxcstr, __truncate_feats(ref_feats, 1), spin=1)[0]
    assert_total_xc_energy(mask, e, n, e_target, n_target, weights, tolerance=1e-7)


gga_cases = {
    'gga_exchange_b88': (gga.e_x_b88, 'gga_x_b88,'),
    'gga_exchange_pbe': (gga.e_x_pbe, 'gga_x_pbe,'),
    'gga_correlation_pbe': (partial(gga.e_c_pbe, zeta=0), ',gga_c_pbe'),  # type: ignore
    'gga_correlation_lyp': (gga.e_c_lyp, ',gga_c_lyp'),
}


@pytest.mark.parametrize('xcfunc,libxcstr', gga_cases.values(), ids=gga_cases.keys())
def test_gga_energy_densities(ethanol_features, xcfunc, libxcstr):
    # FIXME: why is the error that large? (still within 0.0X mHartree)
    weights, mask, feats, ref_feats = ethanol_features
    n, _, s, _ = feats
    e = xcfunc(n=n, s=s)
    n_target = __ref_feats_to_n(ref_feats)
    e_target = dft.libxc.eval_xc(libxcstr, __truncate_feats(ref_feats, 4), spin=1)[0]
    assert_total_xc_energy(mask, e, n, e_target, n_target, weights, tolerance=1e-5)


mgga_cases = {
    'scan': (mgga.e_xc_scan, 'scan'),
}


@pytest.mark.parametrize('xcfunc,libxcstr', mgga_cases.values(), ids=mgga_cases.keys())
def test_mgga_energy_densities(ethanol_features, xcfunc, libxcstr):
    weights, mask, feats, ref_feats = ethanol_features
    n = feats[0]
    e = xcfunc(*feats)
    n_target = __ref_feats_to_n(ref_feats)
    e_target = dft.libxc.eval_xc(libxcstr, ref_feats, spin=1)[0]
    assert_total_xc_energy(mask, e, n, e_target, n_target, weights, tolerance=1e-7)


hybrid_cases = {
    'hf_x': (hybrid.HybridType.HF_X, 'hf,'),
    'pbe0': (hybrid.HybridType.PBE0, 'pbe0'),
    'b3lyp': (hybrid.HybridType.B3LYP, 'b3lyp'),
}


@pytest.mark.parametrize(
    'xctype,libxcstr', hybrid_cases.values(), ids=hybrid_cases.keys()
)
def test_hybrid_functionals(xctype, libxcstr):
    spin_restricted = True
    module = XCModule(
        hybrid.Hybrid(xctype, ElectRepTensorType.EXACT, spin_restricted),
        DensityFeatures(spin_restricted=spin_restricted),
    )
    sys = examples.get('h2o', BASIS, alignment=1)
    pyscf_sys = PySys(
        sys,
        BASIS,
        xc=libxcstr,
        grid_level=1,
        spin_restricted=spin_restricted,
    )
    eri = pyscf_sys.electron_repulsion_tensor
    P = pyscf_sys.density_matrix
    module.init(jax.random.PRNGKey(0), P, sys.grid, eri_tensor=eri)
    E_tot = module.apply({}, P, sys.grid, eri_tensor=eri)
    E_ref = pyscf_sys.xc_energy
    assert abs(E_tot - E_ref) / abs(E_ref) < 1e-6, f'E_tot: {E_tot}, E_ref: {E_ref}'  # type: ignore
