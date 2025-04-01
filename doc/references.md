# References
## General:
*   PyScf: Sun et al. "Recent Developments in the PySCF Program Package." The Journal of Chemical Physics 2020, 153 (2), 024109. https://doi.org/10.1063/5.0006074.
*   libXC: Lehtola, S.; Marques, M. A. L. Reproducibility of Density Functional Approximations: How New Functionals Should Be Reported. The Journal of Chemical Physics 2023, 159 (11), 114116. https://doi.org/10.1063/5.0167763.
*   Psi4: TODO: should I add this?
*   MESS: Helal, H.; Fitzgibbon, A. MESS: Modern Electronic Structure Simulations. arXiv June 5, 2024. https://doi.org/10.48550/arXiv.2406.03121.

## Discretization
### Integrals
Implementation follows MESS (see general references above), which in turn cites:
* Taketa, H.; Huzinaga, S.; & O-ohata, K.; "Gaussian-expansion methods for molecular integrals" Journal of the physical society of Japan 1966, https://doi.org/10.1143/JPSJ.21.2313
* Augspurger JD, Dykstra CE. General quantum mechanical operators. "An open-ended approach for one-electron integrals with Gaussian bases" Journal of computational chemistry. 1990 Jan;11(1):105-11. https://doi.org/10.1002/jcc.540110113
### Density Fitting
TODO:

## Solver
### Self-Consistent-Field (SCF) Method
Follows Susi Lehtola et al. "An Overview of Self-Consistent Field Calculations Within Finite Basis Sets" Molecules 2020, 25 (5), 1218. https://doi.org/10.3390/molecules25051218.
#### Initial Guess

#### Convergence Acceleration / Stabalization
* Direct inversion of the iterative subspace (DIIS):  TODO:


## XC-Energy
* TODO: add KS-paper
### Functionals
#### Classical
* LDA:
    * Perdew-Zunger correlation energy density. Perdew, J. P.; Zunger, "A. Self-Interaction Correction to Density-Functional Approximations for Many-Electron Systems" Phys. Rev. B 1981, https://doi.org/10.1103/PhysRevB.23.5048.
    * VWN5 correlation energy density. Vosko, S. H.; Wilk, L.; Nusair, M. "Accurate Spin-Dependent Electron Liquid Correlation Energies for Local Spin Density Calculations: A Critical Analysis" Can. J. Phys. 1980, https://doi.org/10.1139/p80-159.
* LSDA:
    *  Perdew-Wang correlation energy density (1992) https://doi.org/10.1103/PhysRevB.45.13244
* GGA:
    * Energy density implementations adapted from MESS (see general References)
    * PBE: TODO: add
    * B88: TODO: add
    * LYP: TODO: add
* MGGA:
    * SCAN Sun et al. (Strongly Constrained and Appropriately Normed Semilocal) meta-GGA functional https://doi.org/10.1103/PhysRevLett.115.036402

* Hybrids:
    * PBE0:
    * B3LYP:

#### Learnable
* Nagai 2020:  Ryo Nagai, Ryosuke Akashi, and Osamu Sugino. "Completing Density Functional Theory by Machine Learning Hidden Messages from Molecules." Npj Computational Materials 6, no. 1 (May 5, 2020): 1–8. https://doi.org/10.1038/s41524-020-0310-0.
* Dick 2021: Sebastian Dick, and Marivi Fernandez-Serra. "Highly Accurate and Constrained Density Functional Obtained with Differentiable Programming" Physical Review B 104, no. 16 (October 12, 2021): L161109. https://doi.org/10.1103/PhysRevB.104.L161109.
* DeepMind 2021: J. Kirkpatrick et al. "Pushing the Frontiers of Density Functionals by Solving the Fractional Electron Problem." Science 2021, 374 (6573), 1385–1389. https://doi.org/10.1126/science.abj6511.
* Nagai 2022: Ryo Nagai, Ryosuke Akashi, and Osamu Sugino. "Machine-Learning-Based Exchange Correlation Functional with Physical Asymptotic Constraints." Phys. Rev. Research 2022, 4 (1), 013106. https://doi.org/10.1103/PhysRevResearch.4.013106. [[**REPOSITORY**](https://github.com/ml-electron-project/NNfunctional)]
