from pyscf import dft, gto
import jax.numpy as jnp
import numpy as onp
import pytest

from egxc.systems import examples
from egxc.utils.typing import PRECISION, FloatNx3, FloatN
from egxc.systems import PreloadSystem, System, Grid
from egxc.discretization.grids import atomic
from egxc.discretization import get_grid_fn, get_gto_basis_fn
from utils import set_jax_testing_config

from typing import Tuple

set_jax_testing_config()


def _pyscf_reference_grid(psys: PreloadSystem, grid_level: int = 1):
    pyscf_mol = System.from_preloaded(psys, grid='').to_pyscf('sto-3g')  # type: ignore
    mf = dft.RKS(pyscf_mol)
    mf.grids.level = grid_level
    mf.grids.build()
    grid_coords = jnp.array(mf.grids.coords, dtype=PRECISION.quadrature)
    weights = jnp.array(mf.grids.weights, dtype=PRECISION.quadrature)
    return grid_coords, weights


def _gen_quad_grid(
    psys: PreloadSystem, level=1, pad_to_align: int = 512
) -> Tuple[FloatNx3, FloatN]:
    grid_fn = get_grid_fn(level, psys.atom_z.toset(), pad_to_align)
    return grid_fn(psys.nuc_pos, psys.atom_z, psys.atom_mask)  # type: ignore


def test_gauss_chebychev():
    def pyscf_gauss_chebyshev(n, *args, **kwargs):
        """Gauss-Chebyshev [JCP 108, 3226 (1998); DOI:10.1063/1.475719) radial grids"""
        ln2 = 1 / onp.log(2)
        fac = 16.0 / 3 / (n + 1)
        x1 = onp.arange(1, n + 1) * onp.pi / (n + 1)
        xi = (n - 1 - onp.arange(n) * 2) / (n + 1.0) + (
            1 + 2.0 / 3 * onp.sin(x1) ** 2
        ) * onp.sin(2 * x1) / onp.pi
        xi = (xi - xi[::-1]) / 2
        r = 1 - onp.log(1 + xi) * ln2
        dr = fac * onp.sin(x1) ** 4 * ln2 / (1 + xi)
        return r, dr

    r_ref, dr_ref = pyscf_gauss_chebyshev(10)
    r, dr = atomic._gauss_chebyshev(10)
    assert onp.allclose(r, r_ref)
    assert onp.allclose(dr, dr_ref)


def test_atomic_grids():
    basis = 'sto-3g'
    sys = examples.get('water', basis)
    pyscf_mol = sys.to_pyscf(basis)
    mf = dft.RKS(pyscf_mol)
    ref_g = mf.grids.gen_atomic_grids(mol=pyscf_mol, level=1)

    g = atomic.generate(set(sys.atom_z.tolist()), level=1)
    assert g[1][0].shape == ref_g['H'][0].shape
    assert g[1][1].shape == ref_g['H'][1].shape
    assert g[8][0].shape == ref_g['O'][0].shape
    assert g[8][1].shape == ref_g['O'][1].shape


def __quadrature(mol: gto.Mole, density_matrix, coords, weights, xc='LDA,VWN'):
    """
    Compute the quadrature of the electron density (net charge)
    and the exchange-correlation energy
    """
    ao = dft.numint.eval_ao(mol, coords)
    rho = dft.numint.eval_rho(mol, ao, density_matrix)
    e_xc = dft.libxc.eval_xc(xc, rho, deriv=0)[0]
    return onp.sum(rho * weights), onp.sum(e_xc * rho * weights)  # type: ignore


@pytest.mark.parametrize('align', [1, 4], ids=['no padding', 'with padding'])
def test_quadrature(align):
    psys = examples.get_preloaded(
        'water', basis='6-31G(2df,p)', include_grid=False, alignment=align
    )
    ref_c, ref_w = _pyscf_reference_grid(psys)
    c, w = _gen_quad_grid(psys)
    # generate trajectory of electron densities
    mol = System.from_preloaded(psys, grid='').to_pyscf('6-31G(2df,p)')  # type: ignore
    mf = dft.RKS(mol)
    mf.xc = 'SCAN'
    P_0 = mf.get_init_guess()
    mf.kernel()
    P = mf.make_rdm1()
    ref_q_0, ref_xc_0 = __quadrature(mol, P_0, ref_c, ref_w)
    q_0, xc_0 = __quadrature(mol, P_0, c, w)
    assert 1 - ref_q_0 / q_0 < 5e-9, f'ref_q_0: {ref_q_0}, q_0: {q_0}'
    assert 1 - ref_xc_0 / xc_0 < 5e-9
    ref_q, ref_xc = __quadrature(mol, P, ref_c, ref_w)
    q, xc = __quadrature(mol, P, c, w)
    assert 1 - ref_q / q < 5e-9
    assert 1 - ref_xc / xc < 5e-9
    assert onp.abs(ref_xc - xc) < 1e-6, 'energy error should be less than 1e-6 Ha'


def test_padding(basis='sto-3g'):
    psys = examples.get_preloaded('h2o', basis=basis, include_grid=False, alignment=4)
    c, w = _gen_quad_grid(psys, pad_to_align=psys.grid_alignment, level=3)  # type: ignore
    assert c.shape[0] % 512 == 0
    assert w.shape[0] % 512 == 0
    basis_fn = get_gto_basis_fn(basis, max_period=2, deriv=1)
    aos, grad_aos = basis_fn(
        c,
        psys.nuc_pos,  # type: ignore
        psys.atom_z.array,  # type: ignore
        psys.atom_mask,  # type: ignore
        psys.periods,  # type: ignore
        psys.max_number_of_basis_fns,
    )
    grid = Grid.create(c, w, aos, grad_aos)
    sys = System.from_preloaded(psys, grid=grid)
    P = psys.initial_density_matrix
    n = jnp.einsum('uv,iu,iv->i', P, aos, aos)
    integral = jnp.sum(n * w)

    P = P[sys.fock_tensors.basis_mask, :][:, sys.fock_tensors.basis_mask]
    mol = sys.to_pyscf(basis)
    ao = dft.numint.eval_ao(mol, c)
    n_ref = dft.numint.eval_rho(mol, ao, P)
    ref_integral = onp.sum(n_ref * w)

    assert onp.abs(integral - ref_integral) < 1e-12, 'integral error should be less than 1e-12'

    # initial guess starts from unnormalized density, which is corrected by the scf solver
    mf = dft.RKS(mol)
    mf.kernel()
    P = mf.make_rdm1()
    n_ref = dft.numint.eval_rho(mol, ao, P)
    ref_integral = onp.sum(n_ref * w)
    charge_error = onp.abs(ref_integral - sys.atom_z.sum())
    assert charge_error < 5e-7, f'charge error at grid level 3 should be less than 5e-7: {charge_error}'