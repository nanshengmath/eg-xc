from typing import Dict, List, Tuple
from warnings import warn

import einops
import jax
import jax.numpy as jnp
import numpy as onp
import pyscf
import pyscf.lib.exceptions
import warnings
from flax import linen as nn
from numpy.typing import ArrayLike
from pyscf import gto

from egxc.utils.constants import ANGSTROM_TO_BOHR, L_MAX
from egxc.utils.typing import (
    BoolA,
    IntA,
    FloatAx3,
    FloatAxG,
    FloatAxNxM_SPH,
    FloatZxG,
    FloatG,
    FloatN,
    FloatNx3,
    FloatNxA,
    FloatNxAx3,
    FloatNxB,
    FloatNxBx3,
    FloatNxC_SPH,
    FloatNxM_SPH,
    CompileStaticInt,
    CompileStaticIntA,
)

from typing import Callable

# mapping from l to cartesian angular momentum tuples (lx, ly, lz)
L_TO_LXLYLZ = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # p
    2: [(1, 1, 0), (0, 1, 1), (1, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)],  # d
    3: [  # f
        (3, 0, 0),  # 0     x^3
        (1, 2, 0),  # 1     x   y^2
        (1, 0, 2),  # 2     x       z^2
        (2, 1, 0),  # 3     x^2 y
        (0, 3, 0),  # 4         y^3
        (0, 1, 2),  # 5         y   z^2
        (2, 0, 1),  # 6     x^2     z
        (0, 2, 1),  # 7         y^2 z
        (0, 0, 3),  # 8             z^3
        (1, 1, 1),  # 9     x   y   z
    ],
}

# fmt: off
CART_SPH_CONTRACTIONS = {
    0: onp.eye(1),
    1: onp.eye(3) * 0.707106781186547524, # type: ignore
    2: onp.array([
            # factors from: https://github.com/sunqm/libcint/blob/master/src/cart2sph.c
            [1.092548430592079070, 0, 0, 0,                0,  0],  # d_xy
            [0, 1.092548430592079070, 0,               0,                0,  0],  # d_yz
            [0, 0, 0, -0.315391565252520002, -0.315391565252520002, 0.630783130505040012],  # d_z^2
            [0, 0, 1.092548430592079070,               0,                0,  0],  # d_xz
            [0, 0, 0, 0.546274215296039535, -0.546274215296039535,  0],  # d_(x^2 - y^2)
        ]) / 2.1850968611841584 ,
    3: onp.array([
            # factors from: https://github.com/sunqm/libcint/blob/master/src/cart2sph.c
            [0, 0, 0, 1.770130769779930531, -0.590043589926643510, 0, 0, 0, 0, 0,], # f_y(3x^2 - y^2)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2.890611442640554055], # f_xyz
            [0, 0, 0, -0.457045799464465739, -0.457045799464465739, 1.828183197857862944, 0, 0, 0, 0], # f_yz^2
            [0, 0, 0, 0, 0, 0, -1.119528997770346170, -1.119528997770346170, 0.746352665180230782, 0],
            [-0.457045799464465739, -0.457045799464465739, 1.828183197857862944, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1.445305721320277020, -1.445305721320277020, 0, 0],
            [0.590043589926643510, -1.770130769779930530, 0, 0, 0, 0, 0, 0, 0, 0,],
        ]) / 8.17588381146625823
}
# fmt: on


def _safe_norm(x, axis=-1, keepdims=True):
    x = (jnp.conj(x) * x).sum(axis=axis, keepdims=keepdims)
    zero_mask = x == 0
    x_safe = jnp.where(zero_mask, 1.0, x)
    x_safe = jnp.sqrt(x)
    return jnp.where(zero_mask, 0.0, x_safe)


def _calc_displacements(n: FloatNx3, a: FloatAx3) -> Tuple[FloatNxAx3, FloatNxA]:
    """
    Computes the displacement vectors between two sets of points and
    their corresponding distances.
    TODO: Test whether recomputation adds relevant compute time
    """
    displacements = n[:, None] - a[None, :]
    distances = _safe_norm(displacements)
    return displacements, distances


class GTOAngular(nn.Module):
    """
    Angular part of GTO basis.
    https://doi.org/10.1002/qua.560540202
    Ordering follows
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    https://pyscf.org/user/gto.html#ordering-of-basis-functions
    https://github.com/pyscf/pyscf/blob/a40064009cd3865bce6315d9f87323340e3f343c/pyscf/lib/gto/grid_ao_drv.c

    TODO: presently we are recomputing the angulars for the same set of points
    for many different basis functions. We should probably reformulate the
    shell module as a shell tensor that is only computed once and then
    contract it with the radial part of the individual basis functions.

    QUESTION: Does the jit compiler recognize this and optimize it automatically?
    """

    l: int  # noqa: E741

    def setup(self) -> None:
        assert self.l <= L_MAX, f'Only up to l={L_MAX} is implemented, but got l={self.l}'
        self.ijk_s = jnp.array(L_TO_LXLYLZ[self.l], dtype=jnp.int32)

    def __call__(self, R: FloatNx3) -> FloatNxM_SPH:
        """
        R: displacement vectors relative to GTO center
        """
        cart_angulars = jnp.power(R[..., None, :], self.ijk_s).prod(axis=-1)
        angulars = self._cartesian_to_real_sph(cart_angulars)
        return angulars

    def _cartesian_to_real_sph(self, cartesian_angulars: FloatNxC_SPH) -> FloatNxM_SPH:
        coeff = CART_SPH_CONTRACTIONS[self.l]
        return jnp.einsum('ij,...j->...i', coeff, cartesian_angulars)


class GTOShellFn(nn.Module):
    """
    GTO shell.
    Cramer, Christopher J. (2004).
    Essentials of computational chemistry : theories and models (2nd ed.).
    Chichester, West Sussex, England: Wiley. p. 167.
    ISBN 9780470091821.
    """

    l: int  # noqa: E741

    def setup(self):
        self.gto_angular = GTOAngular(self.l)

    def __call__(
        self,
        displacement: FloatNxAx3,
        distance: FloatNxA,
        ctr_coeffs: FloatAxG,
        basis_exponents: FloatAxG,
    ) -> FloatAxNxM_SPH:
        out = jax.vmap(self._single_atom, in_axes=(1, 1, 0, 0))(
            displacement, distance, ctr_coeffs, basis_exponents
        )
        return out

    def _single_atom(
        self,
        displacement: FloatNx3,
        distance: FloatN,
        ctr_coeffs: FloatG,
        basis_exponents: FloatG,
    ) -> FloatNxM_SPH:
        rnorms = (2 * basis_exponents / jnp.pi) ** (3 / 4) * (8 * basis_exponents) ** (
            self.l / 2
        )
        exps = rnorms[None] * jnp.exp(-(distance**2) * basis_exponents[None])  # FloatNxG
        radials = (ctr_coeffs * exps).sum(-1, keepdims=True)
        angulars = self.gto_angular(displacement)
        return angulars * radials


ShellFn = Callable[[FloatNxAx3, FloatNxA, FloatAxG, FloatAxG], FloatAxNxM_SPH]


def get_gto_shell_fn(angular_momentum: int) -> ShellFn:
    """
    GTO shell.
    Cramer, Christopher J. (2004).
    Essentials of computational chemistry : theories and models (2nd ed.).
    Chichester, West Sussex, England: Wiley. p. 167.
    ISBN 9780470091821.
    """

    assert (
        angular_momentum <= L_MAX
    ), f'Only up to l={L_MAX} is implemented, but got l={angular_momentum}'
    ijk_s = jnp.array(L_TO_LXLYLZ[angular_momentum], dtype=jnp.int32)

    def cartesian_to_real_sph(cartesian_angulars: FloatNxC_SPH) -> FloatNxM_SPH:
        coeff = CART_SPH_CONTRACTIONS[angular_momentum]
        return jnp.einsum('ij,...j->...i', coeff, cartesian_angulars)

    def gto_angular(displacement: FloatNx3) -> FloatNxM_SPH:
        """
        Angular part of GTO basis.
        https://doi.org/10.1002/qua.560540202
        Ordering follows
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        https://pyscf.org/user/gto.html#ordering-of-basis-functions
        https://github.com/pyscf/pyscf/blob/a40064009cd3865bce6315d9f87323340e3f343c/pyscf/lib/gto/grid_ao_drv.c

        TODO: presently we are recomputing the angulars for the same set of points
        for many different basis functions. We should probably reformulate the
        shell module as a shell tensor that is only computed once and then
        contract it with the radial part of the individual basis functions.

        QUESTION: Does the jit compiler recognize this and optimize it automatically?
        """
        cart_angulars = jnp.power(displacement[..., None, :], ijk_s).prod(axis=-1)
        angulars = cartesian_to_real_sph(cart_angulars)
        return angulars

    def single_atom(
        displacement: FloatNx3,
        distance: FloatN,
        ctr_coeffs: FloatG,
        basis_exponents: FloatG,
    ) -> FloatNxM_SPH:
        rnorms = (2 * basis_exponents / jnp.pi) ** (3 / 4) * (8 * basis_exponents) ** (
            angular_momentum / 2
        )
        exps = rnorms[None] * jnp.exp(-(distance**2) * basis_exponents[None])  # FloatNxG
        radials = (ctr_coeffs * exps).sum(-1, keepdims=True)
        angulars = gto_angular(displacement)
        return angulars * radials

    return jax.vmap(single_atom, in_axes=(1, 1, 0, 0))


BasisFn = Callable[
    [FloatNx3, FloatAx3, IntA, BoolA, CompileStaticIntA, CompileStaticInt],
    FloatNxB | Tuple[FloatNxB, FloatNxBx3],
]


def get_gto_basis_fn(string: str, max_period: int, deriv: int) -> BasisFn:
    if max_period >= 5:
        warnings.warn(
            f'Only up to period 5 is supported, but got max_period={max_period},'
            + 'mind that relativistic effects are not considered'
        )

    def atm(z: int) -> gto.Mole | None:
        """
        generate a pyscf molecule with a single atom of atomic number Z
        """
        try:
            return gto.M(atom=[(z, (0, 0, 0))], basis=string, spin=z % 2)
        except pyscf.lib.exceptions.BasisNotFoundError:
            warn(f'Warning: basis {string} not found for Z={z} in pyscf')
            return None

    p_to_bias = {1: 0, 2: 2, 3: 10, 4: 18, 5: 36, 6: 54, 7: 86}
    atoms = [atm(z) for z in range(1, p_to_bias[max_period + 1] + 1)]
    p_to_angulars: Dict[
        int, List[int]
    ] = {}  # i-th element of the list holds the angular momentum of the i-th atomic orbital
    p_to_ctr_coeffs: Dict[
        int, Tuple[FloatZxG]
    ] = {}  # i-th element of the tuple holds contraction coefficients for the i-th atomic orbital
    p_to_exponents: Dict[
        int, Tuple[FloatZxG]
    ] = {}  # i-th element of the tuple holds exponents for the i-th atomic orbital

    def init_exponent_and_ctr_coeff(
        p: int, nao: int, atoms: List[gto.Mole | None], first_non_none: int
    ) -> None:
        """
        initialize the exponents and contraction coefficients for the atomic orbitals in period p
        where the i-th element of the list holds exponents/contraction coefficients for the i-th
        atomic orbital which are of the shape (n_elements_in_period, n_contractions)
        """

        def _nan_save_ctr(i: int, a: gto.Mole | None, template: ArrayLike) -> ArrayLike:
            return template if a is None else a.bas_ctr_coeff(i).reshape(-1)  # type: ignore

        def _nan_save_exp(i: int, a: gto.Mole | None, template: ArrayLike) -> ArrayLike:
            return template if a is None else a.bas_exp(i)

        ctr_coeff = []
        exponents = []
        for i in range(nao):
            template = onp.zeros_like(atoms[first_non_none].bas_exp(i))  # type: ignore
            ctr_coeff.append(
                jnp.stack([_nan_save_ctr(i, a, template) for a in atoms])  # type: ignore
            )
            exponents.append(
                jnp.stack([_nan_save_exp(i, a, template) for a in atoms])  # type: ignore
            )
        p_to_ctr_coeffs[p] = tuple(ctr_coeff)
        p_to_exponents[p] = tuple(exponents)

    max_l = 0

    for p in range(1, max_period + 1):
        atoms_in_period = atoms[p_to_bias[p] : p_to_bias[p + 1]]
        first_non_none = next(
            (i for i, a in enumerate(atoms_in_period) if a is not None), None
        )
        assert first_non_none is not None, f'No atom found for period {p}'
        nao = atoms_in_period[first_non_none].nbas  # type: ignore # number of atomic orbitals of element in period p
        p_to_angulars[p] = [
            atoms_in_period[first_non_none].bas_angular(i)  # type: ignore
            for i in range(nao)
        ]
        # same for all atoms in the same period
        init_exponent_and_ctr_coeff(p, nao, atoms_in_period, first_non_none)
        max_l = max(max_l, max(p_to_angulars[p]))
    l_to_shell_fn = [get_gto_shell_fn(l) for l in range(max_l + 1)]  # noqa: E741
    # assign dicts to frozen member fields
    p_to_angulars = p_to_angulars
    p_to_ctr_coeffs = p_to_ctr_coeffs
    p_to_exponents = p_to_exponents

    def m_sph(angular_momenta: List[int]) -> int:
        # total number of spherical harmonics for a given list of angular momenta
        return sum([2 * L + 1 for L in angular_momenta])

    p_to_aos_per_atom = {p: m_sph(p_to_angulars[p]) for p in range(1, max_period + 1)}

    def aos(
        grid: FloatNx3,
        nuc_pos: FloatAx3,
        atom_z: IntA,
        atom_mask: BoolA,
        periods: CompileStaticIntA,
        max_number_of_basis_fns: CompileStaticInt,
    ) -> FloatNxB:
        """
        Returns the atomic orbitals evaluated at the grid points, with the padded atomic orbitals
        last in the basis dimension.
        """
        # TODO: check if grid is already in bohr
        displacements, distances = _calc_displacements(grid, nuc_pos * ANGSTROM_TO_BOHR)
        out = []
        sort_masks = []
        for p in range(1, max_period + 1):
            p_idx = onp.array(periods) == p
            # TODO: can I check wether p_idx is empty?
            if onp.any(p_idx):
                ao_values = []
                p_disp = displacements[:, p_idx]
                p_dist = distances[:, p_idx]
                Z_idx = (
                    atom_z[p_idx] - 1 - p_to_bias[p]
                )  # TODO: what about Z=0 padding atoms?
                ctr_coeffs = p_to_ctr_coeffs[p]
                p_ctr_coeffs: Tuple[FloatAxG] = jax.tree.map(
                    lambda x: x[Z_idx], ctr_coeffs
                )
                exponents = p_to_exponents[p]
                p_exponents: Tuple[FloatAxG] = jax.tree.map(lambda x: x[Z_idx], exponents)

                for i, l in enumerate(p_to_angulars[p]):  # noqa: E741
                    ao = l_to_shell_fn[l](p_disp, p_dist, p_ctr_coeffs[i], p_exponents[i])
                    ao_values.append(ao)

                ao_values = jnp.concatenate(ao_values, axis=-1)
                # flatten atom dimension into basis dimension
                ao_values = einops.rearrange(
                    ao_values, 'atoms grid basis -> grid (atoms basis)'
                )
                mask = einops.repeat(
                    atom_mask[p_idx],
                    'atoms -> (atoms basis)',
                    basis=p_to_aos_per_atom[p],
                )
                sort_masks.append(mask)
                out.append(ao_values)

        sort_mask = jnp.concatenate(sort_masks, axis=-1)
        order = jnp.argsort(
            sort_mask, stable=True, descending=True
        )  # [0, 1, 1, 0, 0] -> [1, 1, 0, 0, 0]
        out = jnp.concatenate(out, axis=-1)
        return out[:, order[:max_number_of_basis_fns]]

    if deriv == 0:
        out = aos
    elif deriv == 1:

        def aos_and_grad_aos(
            grid: FloatNx3,
            nuc_pos: FloatAx3,
            atom_z: IntA,
            atom_mask: BoolA,
            periods: CompileStaticIntA,
            max_number_of_basis_fns: CompileStaticInt,
        ) -> Tuple[FloatNxB, FloatNxBx3]:
            """
            stucture.period needs to be treated as a static argument w.r.t. jax.jit
            """
            dr_xyz = jnp.eye(3, dtype=grid.dtype)
            ao, grad_ao = jax.vmap(  # vmap over grid point n in N
                lambda r_n: jax.vmap(  # vmap over spatial dimensions x, y, z
                    lambda dr: jax.jvp(
                        lambda x: aos(
                            x[None],
                            nuc_pos,
                            atom_z,
                            atom_mask,
                            periods,
                            max_number_of_basis_fns,
                        )[0],
                        (r_n,),
                        (dr,),
                    ),
                    out_axes=(None, -1),
                )(dr_xyz),
                out_axes=(0, 0),
            )(grid)
            return ao, grad_ao

        out = aos_and_grad_aos
    else:
        raise ValueError(f'Derivative order {deriv} not supported')

    return jax.jit(out, static_argnames=('periods', 'max_number_of_basis_fns'))
