# Density Features
Here we collect some central definitions of (semi-)local electron density features
## Local (Spin) Density Approximation L(S)DA
* electron density $n = n_\text{up} + n_\text{down}$
* spin polarization $\zeta = \frac{n_\text{up} - n_\text{down}}{n}$
* wiegner seitz radius $r_s = \left( \frac{3}{4 \pi n} \right)^{1 / 3}$
## Generalized Gradient Approximation (GGA)
* reduced density gradient $s = \frac{|\nabla n|}{2 (3 \pi^2)^{1/3} n^{4 / 3}}$
## meta-GGA
* kinetic energy density $\tau = \sum_{i, \sigma} \text{occ}(i, \sigma) \frac{|\nabla \psi_{i, \sigma}|^2}{2}$
* Weizsäcker kinetic energy density $\tau_w = \frac{|\nabla n|^2}{8 n}$
* Spin scaling factor for the kinetic energy density $d(\zeta) = [(1 + \zeta)^{5 / 3} + (1 - \zeta)^{5 / 3}] / 2$
* Uniform electron gas kinetic energy density $\tau_\text{unif} = \frac{3}{10} (3 \pi^2)^{2/3} n^{5 / 3} d(\zeta)$
* $\alpha = \frac{\tau - \tau_w}{\tau_\text{unif}}$
