from enum import Enum, unique
import jax.numpy as jnp
from egxc.xc_energy.functionals.base import (
    BaseEnergyFunctional,
    e_x_uniform_electron_gas,
)
from egxc.xc_energy.functionals.classical import lsda, gga
from typing import Callable, NamedTuple, Any
from egxc.utils.typing import (
    Float1,
    FloatN,
    FloatBxB,
    Float2xBxB,
    FloatQxBxB,
    FloatBxBxBxB,
    ElectRepTensorType,
    PRECISION,
)


def e_xc_pbe0(n: FloatN, zeta: FloatN, s: FloatN, **unused_kwargs) -> FloatN:
    return 0.75 * gga.e_x_pbe(n, s) + gga.e_c_pbe(n, zeta, s)


def e_xc_b3lyp(n: FloatN, zeta: FloatN, s: FloatN, **unused_kwargs) -> FloatN:
    """
    The exchange-correlation energy density of B3LYP functional.
    TODO: check whether spin polarization is assumed to be zero.
    """
    e_x = 0.08 * e_x_uniform_electron_gas(n) + 0.72 * gga.e_x_b88(n, s)
    vwn_c = (1 - 0.81) * lsda.vwn5_correlation_energy_density(n, zeta, use_rpa=True)  # type: ignore
    lyp_c = 0.81 * gga.e_c_lyp(n, s)
    b3lyp_xc = e_x + vwn_c + lyp_c
    return b3lyp_xc


class _HybridTypeMember(NamedTuple):
    func: Callable[..., FloatN]
    fraction: float


@unique
class HybridType(Enum):
    HF_X = _HybridTypeMember(lambda *feats, **kwargs: 0, 1.0)  # type: ignore
    PBE0 = _HybridTypeMember(e_xc_pbe0, 0.25)
    B3LYP = _HybridTypeMember(e_xc_b3lyp, 0.2)


def exact_exchange(
    density_matrix: FloatBxB | Float2xBxB, eri_tensor: FloatBxBxBxB, spin_restricted: bool
) -> Float1:
    if spin_restricted:
        P = 0.5 * density_matrix
        out = 2 * jnp.einsum('ijkl,ik,jl', eri_tensor, P, P)
    else:
        Ps = density_matrix
        out = jnp.einsum('ijkl,sik,sjl', eri_tensor, Ps, Ps)
    return -0.5 * out


def density_fitted_exact_exchange(
    density_matrix: FloatBxB | Float2xBxB, df_tensor: FloatQxBxB, spin_restricted: bool
) -> Float1:
    if spin_restricted:
        P = 0.5 * density_matrix
        out = 2 * jnp.einsum('Pij,Pkl,ik,jl', df_tensor, df_tensor, P, P)
    else:
        Ps = density_matrix
        out = jnp.einsum('Pij,Pkl,sik,sjl', df_tensor, df_tensor, Ps, Ps)
    return -0.5 * out


class Hybrid(BaseEnergyFunctional):
    """
    A hybrid exchange-correlation functional, meaning composite functionals
    of local xc energy densities with a fraction of "exact" hatree-fock exchange.
    """
    hybrid_type: HybridType
    eri_type: ElectRepTensorType
    spin_restricted: bool
    is_hybrid = True
    is_graph_based = False

    def setup(self):
        super().setup()
        self.exact_exchange_fraction = self.hybrid_type.value.fraction
        self.e_xc_fn: Callable[..., FloatN] = self.hybrid_type.value.func

    def __call__(
        self, weights: FloatN, *feats: FloatN, **non_local_kwargs: Any
    ) -> Float1:
        """
        overwrites the base class method to include the non-local contributions
        due to exact exchange.
        """
        E_loc = self.integrate_energy_density(weights, *feats)
        E_glob = self.non_local_contribution(**non_local_kwargs)
        assert E_loc.dtype == PRECISION.xc_energy
        assert E_glob.dtype == PRECISION.xc_energy
        return E_loc + E_glob

    def xc_energy_density(
        self, *feats: FloatN, **kwargs: FloatN
    ) -> FloatN:
        # append feats to kwargs
        map_feats = ('n', 'zeta', 's', 'tau')
        for i, f in enumerate(feats):
            kwargs[map_feats[i]] = f  # type: ignore
        return self.e_xc_fn(**kwargs)  # type: ignore

    def non_local_contribution(
        self, density_matrix: FloatBxB | Float2xBxB, eri_tensor: FloatBxBxBxB | FloatQxBxB
    ) -> Float1:
        if self.eri_type == ElectRepTensorType.EXACT:
            E_HFx = exact_exchange(density_matrix, eri_tensor, self.spin_restricted)
        elif self.eri_type == ElectRepTensorType.DENSITY_FITTED:
            E_HFx = density_fitted_exact_exchange(
                density_matrix, eri_tensor, self.spin_restricted
            )
        else:
            raise ValueError(f'Invalid eri_type: {self.eri_type}')
        return self.exact_exchange_fraction * E_HFx
