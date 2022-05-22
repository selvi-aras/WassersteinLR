# WassersteinLR
Source Codes and Datasets used for the paper _Wasserstein Logistic Regression with Mixed Features_ by _Aras Selvi, Mohammad Reza Belbasi, Martin Haugh, and Wolfram Wiesemann (2022)_.

The preprint is available on [Optimization Online](http://www.optimization-online.org/) and [arXiv](https://arxiv.org/). 

## Introduction
This repository provides the following:
- Example scripts that generate synthetic data and apply train:test split. Similarly, scripts that prepare and read UCI datasets.
- The implementation of the (intractable) monolithic exponential-cone optimization problems for Wasserstein logistic regression.
- The implementation of the proposed cutting-plane methods, both for all-categorical and mixed-feature datasets. This includes algorithmic enhancements such as *removing* redundant constraints periodically.
- Cross-validation implementations.
- Examples on a full scheme of tuning parameters via cross-validation, trainin the model on the whole training set, and then report the test set error.

## Dependencies
Our Python3 code to prepare UCI datasets in a format that is compatible with our paper as well as Julia codes uses `pandas` and `NumPy` packages.

The Julia version we used was the [stable release](https://julialang.org/downloads/#current_stable_release) (as of May 2022) v1.7.2. We used [MOSEK](https://www.mosek.com/downloads/) version 9.3 as a conic optimization solver. 

Our Julia codes use the latest (as of May 2022) versions of the following packages: `JuMP` (to call MOSEK and pass the conic problem), `LinearAlgebra` (for norm operations), `Random` (to fix random seeds), `MosekTools` (as a MOSEK interface), `MathOptInterface` (base for JuMP), `Dualization` (for when we solve the dual-conic problem), `DelimitedFiles` (to write CSV files), `InvertedIndices` (for cross-validation), `JLD2` (to save Julia variables), `Statistics` (to get, e.g., quantiles), `DataFrames` (to work with dataframes), `Plots` (Julia's plotting environment), `PlotlyJS` (Julia support of Plotly), `GR` (support visualizations). 

## Description -- All Categorical Datasets
The following is a guide to the scripts related to categorical datasets. Here we include short descriptions and example usage of each script.

<details>
  <summary> <b> generate_data.jl (synthetic data generation, out-of-sample evaluations) </b> </summary>
  
  Calling `generate_dataset(N, n)` returns a training set with `N` rows and `n` binary features, and a test set with `100N` rows and `n` binary features. To construct this data, we first construct a 'true' unit coefficient vector (*i.e.*, true betas and the intercept) at random. Then, we generate the $\pm 1$ predictors at random, and for each instance, we are finding the probability of that instance belonging to label $+1$. The label is then sampled via a Bernuolli distribution according to this probability.
  
  The function `train_test_split(X_raw,y_raw, split)` splits the given dataset `X_raw, y_raw` via a 80\%:20\% train:test ratio. The input `split` is an integer between $1$ and $20$ as in the paper we are randomly splitting UCI datasets $20$ times, and this number drives the random seed.
  
  For a given hypothesis --`beta` (coefficients) and `beta0` (intercept)-- calling `misclassification(X_test, y_test, beta, beta0)` returns the number of misclassified instances on the test set `X_test, y_test` using the hypothesis.
  
</details>

## Final Notes
The following scripts are also available upon request:
- Parallelized codes for the experiments to run on Linux-based cluster computers (we work with [Imperial College Cluster Computers](https://www.imperial.ac.uk/computational-methods/hpc/)).
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
