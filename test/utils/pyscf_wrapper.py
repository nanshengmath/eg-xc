import numpy as onp
from enum import Enum, unique
from pyscf import dft

from egxc.xc_energy.features import transform_abs_grad_n_to_s
from egxc.systems import System
from typing import Tuple, Literal

from numpy.typing import ArrayLike

FloatN = onp.ndarray
FloatNx3 = onp.ndarray
FloatBxB = onp.ndarray
Float2xBxB = onp.ndarray
FloatBxBxBxB = onp.ndarray

FeatureFormat = Literal['egxc', 'pyscf']


@unique
class XcType(Enum):
    lda = 'LDA'
    gga = 'GGA'
    mgga = 'mGGA'


class PyscfSystemWrapper:
    """
    A wrapper around a PySCF system object that provides a convenient interface
    to obtain reference values for different system related properties.
    This wrapper aims to avoid redundant calculations and provide a consistent
    set of features that can serve as reference values for testing.
    """

    __density_matrix = None
    __quad_points_and_weights = None

    def __init__(
        self,
        sys: System,
        basis: str = '6-31G(d)',
        xc: str = 'LDA,VWN',
        spin_restricted: bool = True,
        grid_level: int = 1,
    ):
        self.mol = sys.to_pyscf(basis=basis)
        if self.mol.spin == 0 and spin_restricted:
            self.mf = dft.RKS(self.mol, xc=xc)
            self.spin_restricted = True
        else:
            assert not spin_restricted
            self.mf = dft.UKS(self.mol, xc=xc)
            self.spin_restricted = False
        self.mf.grids.level = grid_level
        self.mf.kernel()

    @property
    def pyscf(self):
        return self.mol

    @property
    def number_of_basis_functions(self) -> int:
        """
        Returns the actual number of basis functions used in the calculation,
        corresponding to the shapes of the density matrix and other related
        tensors.
        """
        return self.mol.nao

    @property
    def total_energy(self) -> float:
        return self.mf.e_tot

    @property
    def one_electron_energy(self) -> float:
        """
        Returns the sum of the kinetic energy of electrons and their attraction
        to nuclei (modelled by the core Hamiltonian).
        """
        elec_energy = self.mf.energy_elec(dm=self.density_matrix)
        return elec_energy[0] - elec_energy[1]

    @property
    def two_electron_energy(self) -> float:
        """
        Returns the electron-electron repulsion energy.
        """
        elec_energy = self.mf.energy_elec(dm=self.density_matrix)
        return elec_energy[1]

    @property
    def xc_energy(self) -> float:
        temp = self.mf.xc
        self.mf.xc = ''
        e_without_xc = self.mf.energy_tot(dm=self.density_matrix)
        self.mf.xc = temp
        return self.total_energy - e_without_xc

    @property
    def nuclear_repulsion_energy(self) -> float:
        return self.mol.energy_nuc()

    @property
    def initial_density_matrix(self) -> FloatBxB:
        return self.mf.get_init_guess()

    @property
    def density_matrix(self) -> FloatBxB | Float2xBxB:
        if self.__density_matrix is None:
            self.__density_matrix = self.mf.make_rdm1()  # type: ignore
        return self.__density_matrix

    @property
    def quadrature_points_and_weights(self) -> Tuple[FloatNx3, FloatN]:
        if self.__quad_points_and_weights is None:
            self.mf.grids.build()
            self.__quad_points_and_weights = self.mf.grids.coords, self.mf.grids.weights  # type: ignore
        return self.__quad_points_and_weights  # type: ignore

    @property
    def overlap(self) -> FloatBxB:
        """
        Returns the overlap matrix of the basis functions.
        """
        return self.mf.get_ovlp()

    @property
    def core_hamiltonian(self) -> FloatBxB:
        return self.mf.get_hcore()

    @property
    def coulomb_matrix(self) -> FloatBxB:
        """
        a.k.a. J matrix
        """
        return self.mf.get_j(dm=self.density_matrix)  # type: ignore

    def xc_potential(self, density_matrix = None) -> FloatBxB:
        if density_matrix is None:
            density_matrix = self.density_matrix
        _, _, v_xc = self.mf._numint.get_vxc(
            self.mol, self.mf.grids, self.mf.xc, dms=density_matrix
        )
        return v_xc

    @property
    def fock_matrix(self) -> FloatBxB:
        return self.mf.get_fock(dm=self.density_matrix)  # type: ignore

    def set_quadrature_points_and_weights(self, point: FloatNx3, weights: FloatN) -> None:
        self.__quad_points_and_weights = point, weights

    @property
    def ao_values(self) -> ArrayLike:
        coords, _ = self.quadrature_points_and_weights
        return self.mol.eval_gto('GTOval_sph_deriv1', coords)

    @property
    def electron_repulsion_tensor(self) -> FloatBxBxBxB:
        return self.mol.intor('int2e')

    def get_density_features(
        self,
        density_matrix: FloatBxB | Float2xBxB | None = None,
        coords: FloatNx3 | None = None,
        xctype: XcType = XcType.mgga,
        format: FeatureFormat = 'pyscf',
    ) -> Tuple[ArrayLike, ...]:
        """
        returns reference density, abs_grad_n and tau
        """
        if density_matrix is None:
            density_matrix = self.density_matrix
        if coords is None:
            ao = self.ao_values
        else:
            ao = dft.numint.eval_ao(self.mol, coords, deriv=1)

        if self.spin_restricted:
            feats = dft.numint.eval_rho(
                self.mol, ao, density_matrix, xctype=xctype.value, with_lapl=False
            )
            if format == 'egxc':
                n = feats[0]
                zeta = onp.zeros_like(n)
                out = (n, zeta)
                if xctype in (XcType.gga, XcType.mgga):
                    abs_grad_n = onp.linalg.norm(feats[1:4], axis=0)
                    s = transform_abs_grad_n_to_s(n, abs_grad_n)
                    out = out + (s,)
                if xctype == XcType.mgga:
                    tau = feats[-1]
                    out = out + (tau,)
            elif format == 'pyscf':
                out = feats
            else:
                raise ValueError('Invalid format')
        else:
            P_up = density_matrix[0]
            P_down = density_matrix[1]
            up_feats = dft.numint.eval_rho(
                self.mol, ao, P_up, xctype=xctype.value, with_lapl=False
            )  # type: ignore
            down_feats = dft.numint.eval_rho(
                self.mol, ao, P_down, xctype=xctype.value, with_lapl=False
            )  # type: ignore
            if format == 'egxc':
                n = up_feats[0] + down_feats[0]
                zeta = (up_feats[0] - down_feats[0]) / n
                out = (n, zeta)
                if xctype in (XcType.gga, XcType.mgga):
                    abs_grad_n = onp.linalg.norm(up_feats[1:4] + down_feats, axis=0)
                    s = transform_abs_grad_n_to_s(n, abs_grad_n)
                    out += (s,)
                if xctype == XcType.mgga:
                    tau = up_feats[-1] + down_feats[-1]
                    out = out + (tau,)
            elif format == 'pyscf':
                out = (up_feats, down_feats)
            else:
                raise ValueError('Invalid format')

        return out  # type: ignore