# WassersteinLR
Source Codes and Datasets used for the paper _Wasserstein Logistic Regression with Mixed Features_ by _Aras Selvi, Mohammad Reza Belbasi, Martin Haugh, and Wolfram Wiesemann (2022)_.

The preprint is available on [Optimization Online](http://www.optimization-online.org/) and [arXiv](https://arxiv.org/). 

## Dependencies
Our Python3 code to prepare UCI datasets in a format that is compatible with our paper as well as Julia codes uses `pandas` and `NumPy` packages.

The Julia version we used was the [stable release](https://julialang.org/downloads/#current_stable_release) (as of May 2022) v1.7.2. We used [MOSEK](https://www.mosek.com/downloads/) version 9.3 as a conic optimization solver. 

Our Julia codes use the latest (as of May 2022) versions of the following packages: `JuMP` (to call MOSEK and pass the conic problem), `LinearAlgebra` (for norm operations), `Random` (to fix random seeds), `MosekTools` (as a MOSEK interface), `MathOptInterface` (base for JuMP), `Dualization` (for when we solve the dual-conic problem), `DelimitedFiles` (to write CSV files), `InvertedIndices` (for cross-validation), `JLD2` (to save Julia variables), `Statistics` (to get, e.g., quantiles), `DataFrames` (to work with dataframes), `Plots` (Julia's plotting environment), `PlotlyJS` (Julia support of Plotly), `GR` (support visualizations). 


## Final Notes
The following scripts are also available upon request:
- Parallelized implementation of all the experiments to run on Linux-based cluster computers.
- [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System) job codes.
- Codes we used to visualize experiments.

## Thank You
Thank you for your interest in our work. If you found this work useful in your research and/or applications, please star this repository and cite:
```
@article{WassersteinLR,
  title={Wasserstein logistic regression with mixed features},
  author={Selvi, Aras and Belasi, Mohammad Reza and Haugh, Martin and Wiesemann, Wolfram},
  journal={Available on Optimization Online},
  year={2022}
}
```
Please contact Aras (a.selvi19@imperial.ac.uk) if you encounter any issues using the scripts. For any other comment or question, please do not hesitate to contact us:

[Aras Selvi](https://www.imperial.ac.uk/people/a.selvi19) _(a.selvi19@imperial.ac.uk)_

[Mohammad Reza Belbasi](https://uk.linkedin.com/in/mohammad-reza-belbasi-5267a512a) _(r.belbasi21@imperial.ac.uk)_

[Martin Haugh](https://martin-haugh.github.io/) _(m.haugh@imperial.ac.uk)_

[Wolfram Wiesemann](http://wp.doc.ic.ac.uk/wwiesema/) _(ww@imperial.ac.uk)_
