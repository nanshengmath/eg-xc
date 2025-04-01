import jax.numpy as jnp
from egxc.systems import examples
from egxc.utils.typing import PRECISION
from utils import set_jax_testing_config

set_jax_testing_config()

def test_system():
    mol = examples.get('ethanol', 'sto-3g', alignment=False)
    # try to create a pyscf molecule
    pyscf_mol = mol.to_pyscf('sto-3g')
    pyscf_mol.kernel()

    assert jnp.ndim(mol.atom_z) == 1, 'Atomic numbers must be 1D'
    assert jnp.ndim(mol._nuc_pos) == 2, 'Positions must be 2D'
    assert (
        mol._nuc_pos.dtype == PRECISION.basis
    ), 'Positions must be in basis precision'
    assert mol.atom_z.dtype == jnp.uint8, 'Atomic numbers must be integers'
    assert jnp.all(
        mol.atom_z[:-1] <= mol.atom_z[1:]
    ), 'Atomic numbers should be ordered by charge'

    assert mol._nuc_pos.shape == (9, 3)

def test_spin_unrestricted_system():
    mol = examples.get('h2+', basis='sto-3g')
    # try to create a pyscf molecule
    pyscf_mol = mol.to_pyscf('sto-3g')
    pyscf_mol.kernel()

    assert jnp.ndim(mol.atom_z) == 1, 'Atomic numbers must be 1D'
    assert jnp.ndim(mol._nuc_pos) == 2, 'Positions must be 2D'
    assert (
        mol._nuc_pos.dtype == PRECISION.basis
    ), 'Positions must be in basis precision'
    assert mol.atom_z.dtype == jnp.uint8, 'Atomic numbers must be integers'
    atom_z = mol.atom_z[mol.atom_mask]
    assert jnp.all(
        atom_z[:-1] <= atom_z[1:]
    ), f'Atomic numbers should be ordered by charge Z = {atom_z}'

    assert mol._nuc_pos.shape == (4, 3)

def test_padded_system():
    # TODO: update to period specific padding
    padded_sys =  examples.get("water", basis='sto-3g', alignment=4)
    assert padded_sys.atom_z.shape == (4,)
    assert padded_sys._nuc_pos.shape == (4, 3)
    assert jnp.all(padded_sys.atom_mask == jnp.array([1, 1, 1, 0], dtype=bool))
