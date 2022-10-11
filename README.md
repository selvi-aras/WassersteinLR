# WassersteinLR
Source Codes and Datasets used for the paper _Wasserstein Logistic Regression with Mixed Features_ by _Aras Selvi, Mohammad Reza Belbasi, Martin Haugh, and Wolfram Wiesemann (2022)_.

The preprint is available on [Optimization Online](http://www.optimization-online.org/DB_HTML/2022/05/8929.html) and [arXiv](https://arxiv.org/abs/2205.13501).

*Update:* This paper has been accepted by **NeurIPS022**.

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

<details>
  <summary> <b> logistic_regressions.jl (non-robust naive or non-robust regularized Logistic Regression) </b> </summary>
  
  For a given training set `X, y`, calling `logistic_regression(X,y, groups; regular = 0, lambda = 0)` returns an optimized `JuMP` model for logistic regression trained on this set. To see the decisions, we call `model_summarize(model)`. Here, `groups` gives us the groups of binary variables that correspond to original categorical features. For example, if we have 4 predictors, we can have `groups = [1:1, 2:2, 3:3, 4:4]`, which means all binary variables are corresponding to original features, whereas `groups = [1:1, 2:4]` means the first binary variable corresponds to an original feature whereas second, third and fourth variables are the dummy variables for an original feature with three possible categories. Moreover, `regular = 0` calls the naive logistic regression, `regular = 1` calls the LASSO-regularized logistic regression, and `regular = 2` calls the Ridge-regularized logistic regression. In case where `regularized` is not zero, one should specify the regularization penalty parameter `lambda`. 
  
</details>

<details>
  <summary> <b> monolithic.jl (Wasserstein Logistic Regression -- monolithic implementation) </b> </summary>
  
  For a given training set `X, y`, as well as the list of dummies for each original feature `groups`, calling `monolithic_wasserstein(X, y, groups, epsilon; regular = 0, pen = 0, dual_conic = 0, metric = 0, p = 1, kappa = 1, restriction =0)` solves the Wasserstein DRO formulation of logistic regression. Here, the input `epsilon` is the radius of the Wasserstein ball. The input parameter `regular` is to set the regularization (0: no regularization, 1: LASSO, 2: Ridge) and in case of regularization the penalty parameter is given via `pen`. The input `dual_conic`, if set to 1, solves the dual exponential conic problem instead of the primal problem. The ground metric is decided by `metric`, and in our paper we always use `metric = 1` which corresponds to the feature-label metric, however, `metric = 0` is a coarse metric that returns $1$ if two instances are identical and $0$ otherwise. In case `metric= 1` is chosen as in our paper, the parameter `p` sets the $p$-norm to take for the dsitance $||x - x'||$ and `kappa` sets the $\kappa$-variable used in our distance metric to weigh the label mismatch.
  
</details>

<details>
  <summary> <b> cutting.jl (Wasserstein Logistic Regression -- cutting-plane implementation) </b> </summary>
  
  The function `cutting_wasserstein_updated` solves the Wasserstein DRO formulation of logistic regression via the cutting-plane method we propose. Needless to say, the output will be the same with the monolithic solution in `monolithic.jl`, though the solution is faster as demonstrated in Figure 2 of our paper. The input of this function is idential with those of `monolithic_wasserstein` in the file `monolithic.jl`. However, the output has an additional value, named `iteration`, standing for number of iterations the cutting-plane algorithm took before termination, as well as `solver_times` that returns the time MOSEK took to solve each sub-problem. 
    
</details>


<details>
  <summary> <b> cross_validate.jl (cross validation for non-robust and distributionally robust logistic regression) </b> </summary>
  
  For a given dataset `X, y`, as well as the list of dummies for each original feature `groups`, calling `cv_logistic_regression(X,y, groups; regular = 0, lambda = 0)` returns a list of five errors, corresponding to 5-CV errors of regularized (LASSO if `regular = 1`, Ridge if `regular = 2`) logistic regression corresponding to the regularization penalty `lambda`. 
  
  Similarly, `cv_wasserstein(X, y, groups, epsilon; regular= 0, pen= 0, dual_conic = 0, metric = 0, p = 1, kappa = 1)` returns the CV errors for the Wasserstein DRO model with given `epsilon` (radius of the Wasserstein ball), `regular` and `pen` (regularization and penalty), `metric, p, kappa` (values defining the ground metric -- note: in our paper we always use `metric = 1, p=1` and try `kappa = 1` or `kappa = T` with `T` being number of binary variables).
    
</details>

<details>
  <summary> <b> main_hpc.jl (An example implementation of the experiments for unregularized methods) </b> </summary>
  
  The script that is called from the high performance computers for parallelized runs for *unregularized* non-robust logistic regression and *unregularized* Wasserstein logistic regression. There is a single function named `hpc(job_nr)` that takes a job number in, figures out the parameter setting we would like to run (*e.g.*, second UCI dataset, Wasserstein DRO, $\epsilon = 0.1$, $p=1$, $\kappa = 1$, third train:test split, etc.), and saves the relevant results via `jldsave` command of `JLD2` package. The so-called "relevant results" include 5-fold CV errors of the specific parameter setting, the optimal beta values over the whole training set (to be taken **if** decided from the validation steps), number of cutting-plane iterations it took over the whole training set, test-set error corresponding to the trained model (again, only to be reported **after** validating a model via cross validation). The set of `job_nr` to give, for example, for the first dataset, are $1-440$, and the second dataset are $441-880$, etc. The explanations can be seen as comments in this script.
    
</details>


<details>
  <summary> <b> interpret_results.jl (Interpret the results of main_hpc.jl) </b> </summary>
  
  At line 17, specifying `d_to_take` value corresponds to which UCI dataset to interpret. Running the script will return, for example, median errors of each method under 20 train:test split for that dataset (includes model selection via cross-validation).
    
</details>

<details>
  <summary> <b> regularized_main_hpc.jl, regularized_interpret_results.jl (Extensions to regularized methods) </b> </summary>
  
  Analogous to main_hpc.jl and interpret_results.jl, respectively, however, extended to regularization. For example, instead of cross-validating the Wasserstein DRO Logistic Regression model for each $\epsilon$, we cross-validate for each pair of $\epsilon$ (Wasserstein ball radius) and $\lambda$ (LASSO penalty).
   
</details>

## Description -- Mixed-Feature Datasets
The following is a guide to the scripts related to mixed-feature datasets. These are simple extensions of the scripts for categorical datasets we presented above. One can only keep the mixed-feature scripts, as these scripts generalize the scripts for all-categorical datasets. 

<details>
  <summary> <b> mixed_generate_data.jl (synthetic data generation, out-of-sample evaluations) </b> </summary>
  
  Analogous to `generate_data.jl`. The only extension is the existence of continuous (numeric) features. The function `mixed_generate_dataset(N, n, n_cont)` as opposed to `generate_dataset(N, n)`, has an additional input `n_cont`, standing for number of continuous feature. Similarly, `mixed_misclassification(X_cont_test, X_test, y_test, beta_cont_opt, beta_opt, beta0_opt)` takes additional inputs `X_cont_test` (a matrix standing for the continuous-features of the test indices), and `beta_cont_opt` (beta coefficients corresponding to the continuous variables). A row of `X_test` and `X_cont_test` correspond to the same instance, however, we collect the binary and continuous variables via separate matrices for the ease of implementation (especially for the cutting-plane method, since the distance calculations of continuous and binary variables differ).

</details>

<details>
  <summary> <b> mixed_logistic_regressions.jl (non-robust naive or non-robust regularized Logistic Regression) </b> </summary>
  
  Analogous to `logistic_regressions.jl`. Compared to the function `logistic_regression`, the function `mixed_logistic_regression` also takes an additional matrix `X_cont` corresponding to the continuous (numeric) variables.
  
</details>

<details>
  <summary> <b> mixed_cutting.jl (Wasserstein Logistic Regression -- cutting-plane implementation) </b> </summary>
  
  Analogous to `cutting.jl`. Compared to the function `cutting_wasserstein`, the function `mixed_logistic_regression` also takes an additional matrix `X_cont` corresponding to the continuous (numeric) variables.
    
</details>


<details>
  <summary> <b> mixed_cross_validate.jl (cross validation for non-robust and distributionally robust logistic regression) </b> </summary>
  
  Analogous to `cross_validate.jl`, and similar to other scripts for mixed features, the functions have additional inputs `X_cont` standing for continuous features of a dataset.
    
</details>

We are omitting the mixed-feature generalization of previous scripts `main_hpc.jl`, `interpret_results.jl`, `regularized_main_hpc.jl`, and `regularized_interpret_results.jl`, since these are mostly identical with the only difference being name of the functions called, and the name of UCI datasets.

## Final Notes
In this repository, we included core codes to illustrate the implementation of our algorithm. However, there are some extension or codes not provided, as they are not directly relevant to our findings. To this end, we would like to note that, the following scripts are also available upon request:
- Parallelized codes for the experiments to run on Linux-based cluster computers (we work with [Imperial College Cluster Computers](https://www.imperial.ac.uk/computational-methods/hpc/)).
- [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System) job codes.
- Codes we used to visualize experiments (via `PlotlyJS` and `Julia.Plots`).
- Some algorithmic techniques on speeding up the cutting-plane method are available (*e.g.*, using hash tables to check if a constraint was deleted before not to cycle).
- Implementation of benchmark methods from the literature, for example, those for the experiments mentioned in Section 2.2.
- Solving nature's optimization problem for each fixed beta to find the worst-case distributions, for example, in order to get the right plot of Figure 1 of our paper.
- *Update:* In our revision, we have included new benchmark algorithms, including Wasserstein Profile Inference. Implementation of this technique will be pushed to this repository before the NeurIPS2022 conference. 

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
