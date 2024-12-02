import jax.numpy as jnp

from egxc.structures import create_structure


def test_structure():
    # instantiate a Helium atom
    mol = create_structure(2, [0, 0, 0], 0, 0)  # type: ignore
    # try to create a pyscf molecule
    pyscf_mol = mol.to_pyscf_mol('sto-3g')
    pyscf_mol.kernel()
    padded_structure = mol.pad_structure(4)

    assert mol.positions.shape == (1, 3)
    assert padded_structure.atomic_numbers.shape == (5,)
    assert padded_structure.positions.shape == (5, 3)
    assert (padded_structure.mask == jnp.array([1, 0, 0, 0], dtype=bool)).all()
