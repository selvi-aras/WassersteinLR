# WassersteinLR
Source Codes and Datasets used for the paper _Wasserstein Logistic Regression with Mixed Features_ by _Aras Selvi, Mohammad Reza Belbasi, Martin Haugh, and Wolfram Wiesemann (2022)_.

The preprint is available on [Optimization Online](http://www.optimization-online.org/) and [arXiv](https://arxiv.org/). 

## Dependencies
Our Python3 code to prepare UCI datasets in a format that is compatible with our paper as well as Julia codes uses `pandas` and `NumPy` packages.

The Julia version we used was the [stable release](https://julialang.org/downloads/#current_stable_release) (as of May 2022) v1.7.2. We used [MOSEK](https://www.mosek.com/downloads/) version 9.3 as a conic optimization solver. 

Our Julia codes use the latest (as of May 2022) versions of the following packages: `JuMP` (to call MOSEK and pass the conic problem), `LinearAlgebra` (for norm operations), `Random` (to fix random seeds), `MosekTools` (as a MOSEK interface), `MathOptInterface` (base for JuMP), `Dualization` (for when we solve the dual-conic problem), `DelimitedFiles` (to write CSV files), `InvertedIndices` (for cross-validation), `JLD2` (to save Julia variables), `Statistics` (to get, e.g., quantiles), `DataFrames` (to work with dataframes), `Plots` (Julia's plotting environment), `PlotlyJS` (Julia support of Plotly), `GR` (support visualizations). 
