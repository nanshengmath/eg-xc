# Learning Equivariant Non-Local Electron Density Functionals

![Title](figures/title.png)

<!-- ⚠️ This Repository is Currently Under Construction ⚠️ -->
<div style="color: red; font-weight: bold; padding: 10px; border: 2px solid red; text-align: center; margin-bottom: 20px;">
⚠️ DISCLAIMER: This repository is currently under construction. Code and documentation may be incomplete or subject to change. ⚠️
</div>

Reference implementation of Equivariant Graph Exchange Correlation (EG-XC) from

*Learning Equivariant Non-Local Electron Density Functionals* <br>
by Nicholas Gao*, Eike Eberhard* and Stephan Günnemann <br>
published as Spotlight at ICLR 2025. <br>
https://openreview.net/forum?id=FhBT596F1X

## Installation
1. Install [`uv`](https://docs.astral.sh/uv/):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Create a virtual environment and install dependencies
    ```sh
    uv sync
    source .venv/bin/activate
    ```

## Contact
Please contact [n.gao@tum.de](mailto:n.gao@tum.de) and [e.eberhard@tum.de](mailto:e.eberhard@tum.de) if you have any questions.


## Cite
Please cite our paper if you use our method or code in your own works:
```
@inproceedings{gao_eberhard_2025_egxc,
    title={Learning Equivariant Non-Local Electron Density Functionals},
    author={Nicholas Gao and Eike Eberhard and Stephan G{\"u}nnemann},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```