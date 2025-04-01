import jax.numpy as jnp
import numpy as onp
import einops
from pyscf import dft

from egxc.discretization import get_grid_fn
from egxc.discretization.basis import get_gto_basis_fn
from egxc.systems import PreloadSystem, System, examples
from utils import relative_error, PyscfSystemWrapper, set_jax_testing_config

set_jax_testing_config()


def check_equal_ao_values(psys: PreloadSystem, deriv: int = 1, max_period: int = 2):
    grid_fn = get_grid_fn(1, psys.atom_z.toset(), 512)  # type: ignore
    sys = System.from_preloaded(psys, grid='')  # type: ignore

    nuc_pos = jnp.asarray(psys.nuc_pos)
    coords, weights = grid_fn(nuc_pos, psys.atom_z, psys.atom_mask)  # type: ignore
    basis_fn = get_gto_basis_fn(psys.basis, max_period=max_period, deriv=deriv)
    args = (
        coords,
        nuc_pos,
        sys.atom_z,
        sys.atom_mask,
        psys.periods,
        psys.max_number_of_basis_fns,
    )

    pyscf_mol = sys.to_pyscf(psys.basis)
    if deriv == 0:
        aos = basis_fn(*args)  # type: ignore
        pyscf_target_aos = pyscf_mol.eval_gto('GTOval_sph', coords)
    elif deriv == 1:
        aos, grad = basis_fn(*args)  # type: ignore
        aos = jnp.concatenate((aos[..., None], grad), axis=-1)  # type: ignore
        pyscf_target_aos = pyscf_mol.eval_gto('GTOval_sph_deriv1', coords)  # type: ignore
        aos = einops.rearrange(aos, 'n b f -> f n b')
    else:
        raise ValueError(f'Derivative order {deriv} not supported')

    aos = aos[..., sys.fock_tensors.basis_mask]  # type: ignore

    assert (
        aos.shape == pyscf_target_aos.shape  # type: ignore
    ), f'Shapes do not match: {aos.shape} != {pyscf_target_aos.shape}'  # type: ignore

    weighted_absolute_error = jnp.abs(aos - pyscf_target_aos) * weights[None, :, None]

    assert (
        weighted_absolute_error.max() < 5e-14
    ), f'Maximum absolute error is too high (abs-err: {weighted_absolute_error.max()}), basis: {psys.basis}'

    assert (
        weighted_absolute_error.sum() < 5e-11
    ), f'Total absolute error is too high (abs-err: {weighted_absolute_error.sum()}), basis: {psys.basis}'

    pyscf_target_aos = jnp.where(abs(pyscf_target_aos) < 1e-15, 0.0, pyscf_target_aos)
    weighted_relative_error = (
        relative_error(aos, pyscf_target_aos, eps=1e-12) * weights[None, :, None]
    )

    assert (
        weighted_relative_error.mean() < 2e-3
    ), f'Mean relative error is too high (rel-err: {weighted_relative_error.mean()}), basis: {psys.basis}'
    assert (
        weighted_relative_error.max() < 2e-1
    ), f'Maximum relative error is too high (rel-err: {weighted_relative_error.max()}), basis: {psys.basis}'


def test_ao_values_single_atom_without_gradient(basis: str = 'sto-3g'):
    psys = examples.get_preloaded('h', basis=basis, include_grid=False, alignment=1)
    check_equal_ao_values(psys, deriv=0)


def test_ao_values_single_atom(basis: str = 'sto-3g'):
    psys = examples.get_preloaded('h', basis=basis, include_grid=False, alignment=1)
    check_equal_ao_values(psys)


def test_ao_values_multiple_atoms(basis: str = 'sto-3g'):
    psys = examples.get_preloaded('organic', basis=basis, include_grid=False, alignment=1)
    check_equal_ao_values(psys)


def test_ao_values_with_padding(basis: str = 'sto-3g', align: int = 4):
    psys = examples.get_preloaded('h2', basis=basis, include_grid=False, alignment=align)
    check_equal_ao_values(psys)


def test_elements_basis_sets(z=18, basis='6-31G(d)'):
    # test first 18 elements
    psys = examples.get_preloaded(z, basis=basis, include_grid=False, alignment=1)
    check_equal_ao_values(psys, max_period=3)


def test_larger_basis_sets():
    for b in [
        'sto-6g',
        '6-31G(d)',
        '6-31G(2df,p)',
        '6-311+G(2df,p)',
        # '6-311++G(3df,2pd)',  FIXME: These two seem to have a problem with
        # 'def2-TZVP',          FIXME: shapes mismatch in the construction of the ctr_coeff
    ]:
        psys = examples.get_preloaded('organic', basis=b, include_grid=False, alignment=1)
        check_equal_ao_values(psys)


def test_combination_of_quadrature_and_atomic_orbitals(
    basis='6-31G(2df,p)', xc='LDA,VWN'
):
    psys = examples.get_preloaded('water', include_grid=False, basis=basis)

    grid_fn = get_grid_fn(1, psys.atom_z.toset(), 512)
    basis_fn = get_gto_basis_fn(psys.basis, max_period=3, deriv=0)

    mol = PyscfSystemWrapper(System.from_preloaded(psys, grid=''), basis=basis, xc=xc)  # type: ignore
    P = mol.density_matrix

    def compute():
        coords, weights = grid_fn(psys.nuc_pos, psys.atom_z, psys.atom_mask)  # type: ignore
        aos = basis_fn(
            coords,
            psys.nuc_pos,  # type: ignore
            psys.atom_z.array,  # type: ignore
            psys.atom_mask,  # type: ignore
            psys.periods,  # type: ignore
            psys.max_number_of_basis_fns,
        )  # type: ignore
        aos = onp.array(aos, dtype=onp.float64)
        rho = dft.numint.eval_rho(mol.pyscf, aos, P)
        e_xc = dft.libxc.eval_xc(xc, rho, deriv=0)[0]
        return e_xc, onp.sum(e_xc * rho * weights)  # type: ignore

    e_xc, E_xc = compute()

    def compute_references():
        ref_c, ref_w = mol.quadrature_points_and_weights
        ref_ao = dft.numint.eval_ao(mol.pyscf, ref_c)
        ref_rho = dft.numint.eval_rho(
            mol.pyscf, ref_ao, P
        )  # TODO: mind that this assumes that the coordinates are in units of Bohr
        ref_e_xc = dft.libxc.eval_xc(xc, ref_rho, deriv=0)[0]
        return ref_e_xc, onp.sum(ref_e_xc * ref_rho * ref_w)  # type: ignore

    ref_e_xc, ref_E_xc = compute_references()

    # TODO: is our implementation really that good?!
    assert 1 - E_xc / ref_E_xc < 1e-15, f'ref_E_xc: {ref_E_xc}, E_xc: {E_xc}'
    assert onp.abs(E_xc - ref_E_xc) < 1e-12, 'energy error should be less than 1e-12 Ha'
