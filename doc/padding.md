# Padding
Since the JAX jit has to recompile for each individual input shape, we add padding.
A system/input consists of $A^{(p)}$ atoms. The total number of basis functions $B$ only depends on
$A^{(p)}$ the number of atoms in each period of the PSE and the number of basis functions
$b^{(p)}$ for each period
$$
    A = \sum_{p=1}^{7} A^{(p)}  \hspace{2cm} B = \sum_{p=1}^{7} b^{(p)} A^{(p)} \: .
$$
While the number of quadrature points $N$ roughly scales with the number of atoms like
$$
    N \approx \textrm{const.} \cdot (A_{p = 1} + 2 A_{p \neq 1}) \: .
$$

In the following we denote padded quantities with a tilde, e.g. $\tilde{A_i}$ for the
number of atoms of the $i$-th system in the dataset. We want to reduce the number of
unique tuples
$$
    (\tilde{A^{(1)}_i}, \dots , \tilde{A^{(P)}_i}, \tilde{B_i}, \tilde{N_i})
$$
in the dataset, whitout adding exessive computational overhead.

If the basis functions are evaluated using jax the number of basis functions $B$ is
determined by the number of atoms per period $A^{(p)}$ and the number of basis functions per atom $B^{(p)}$,
hence
$$
    \tilde{B_i} = \sum_{p=1}^{7} b^{(p)}_i \tilde{A^{(p)}_i} \: .
$$
If we want to enforce an additional



## Additional Memory Considerations
Since the memory requirement of the electron repulsion tensor scales with $\mathcal{O}(\tilde{B}^4)$ we want to keep $\tilde{B}$ in particular as small as possible.