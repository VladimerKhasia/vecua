<div align="center">

# The Adaptive Vekua Cascade: A Differentiable Spectral-Analytic Solver for Physics-Informed Representation


</div>

<p align="center">
  <strong>Implementation of the paper "The Adaptive Vekua Cascade: A Differentiable Spectral-Analytic Solver for Physics-Informed
Representation"</strong>
</p>


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17916218.svg)](https://doi.org/10.5281/zenodo.17916218)



## Abstract

Coordinate-based neural networks have emerged as a powerful tool
for representing continuous physical fields, yet they face two fundamental
pathologies: spectral bias, which hinders the learning of high-frequency
dynamics, and the curse of dimensionality, which causes parameter explo-
sion in discrete feature grids. We propose the Adaptive Vekua Cascade
(AVC), a hybrid architecture that bridges deep learning and classical
approximation theory. AVC decouples manifold learning from function
approximation by using a deep network to learn a diffeomorphic warp-
ing of the physical domain, projecting complex spatiotemporal dynamics
onto a latent manifold where the solution is represented by a basis of
generalized analytic functions. Crucially, we replace the standard gradient-
descent output layer with a differentiable linear solver, allowing the
network to optimally resolve spectral coefficients in a closed form during
the forward pass. We evaluate AVC on a suite of five rigorous physics
benchmarks, including high-frequency Helmholtz wave propagation, sparse
medical reconstruction, and unsteady 3D Navier-Stokes turbulence. Our
results demonstrate that AVC achieves state-of-the-art accuracy while
reducing parameter counts by orders of magnitude (e.g., 840 parame-
ters vs. 4.2 million for 3D grids) and converging 2-3× faster than
implicit neural representations. This work establishes a new paradigm for
memory-efficient, spectrally accurate scientific machine learning.
---

### Experiment Manifest

To reproduce the results, run `AVC.py` or for more convenience turn it into Jupyter Notebook `AVC.ipynb`. 


---

## Installation

1. For quick experimentation with Jupyter Notebook

```bash
pip install jax optax numpy pandas matplotlib  
```
2. Or clone the repository to your local machine and install the required dependencies using pip:

```bash
pip install -r requirements.txt
# Once the dependencies are installed, you can execute the script using:
python AVC.py
```


## Citation

If you utilize this code or the concepts presented in **AVC** for your research, please cite the following paper:

```bibtex
@misc{khasia2025avc_zenodo,
  author       = {Khasia, Vladimer},
  title        = {The Adaptive Vekua Cascade: A Differentiable Spectral-Analytic Solver for Physics-Informed Representation},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17916218},
  url          = {https://doi.org/10.5281/zenodo.17916218},
  note         = {Preprint}
}
```







