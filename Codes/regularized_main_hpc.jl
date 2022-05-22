##Packages used
using JuMP #to call solvers and formulate optimization problems
using LinearAlgebra #to use functions like 'norm'
import Random #for random number generation
import MosekTools #MOSEK
import MathOptInterface #used by JuMP
const MOI = MathOptInterface #referring to MathOptInterface simply as 'MOI'
using MAT #if you want to read data from MATLABx
using Dualization
using DelimitedFiles #CSV, DataFrames,
using InvertedIndices
using JLD2 #to save files to save: jldsave("Data/savedata"; beta_opt_metric, beta0_opt_metric, groups), to load
#obj = load("Data/savedata"), and then call a, e.g., matrix by obj["beta_opt_metric"]
#Include functions we wrote and generate data
include("./logistic_regressions.jl") #all logistic regression functions
include("./generate_data.jl") #for generating data of a problem
include("./monolithic.jl") #Wasserstein approach with monolithic solution
include("./cutting.jl") #Cutting plane algorithms
include("./cross_validate.jl") #the functions that return CV'ed error for a given model
function hpc_reg(job_nr)
    ##now iterate over all the parameter combinations
    #************JOB NR MUST BE GIVEN
    #job_nr = 1
    #
    dataset_index = 1:18 #there are 18 datasets
    max_split = 20 #20 tr/te splits
    lr_wasser = ["lr", "wasser"]
    metric_index = [0 1] #0 is coarse, 1 is the good one
    epsilon_index = 1:5 #5 epsilons will be tried
    lambda_index = 1:7 #7 lambdas for regularized-logistic regression
    regularized_lambda_index = 1:5 #5 lambdas for regularized Wasserstein DRO
    kappa_index = 1:2 #two kappas will be tried
    iteration_hpc = 0 #running counter

    # will return the following to uniquely identify all the parameters to run
    d_to_take = 1 #dataset index to use later
    split_to_take = 1 #dataset split to use later
    lr_wasser_to_use = "lr"
    metric_to_use = 0
    epsilon_to_use = 1
    lambda_to_use = 1
    kappa_to_use  = 1
    #read parameters
    for d in dataset_index #iterate over dataset
        for split_nr in 1:max_split #iterate over splits
            for meth in lr_wasser #iterate over all methods
                if meth == "lr" #if we are taking the logistic regression (regularized)
                    for lambda in lambda_index
                        iteration_hpc = iteration_hpc +  1 #A Method is being called
                        if iteration_hpc == job_nr #********************
                            d_to_take = copy(d)
                            split_to_take = copy(split_nr)
                            lr_wasser_to_use = meth
                            lambda_to_use = copy(lambda)
                        end
                    end
                elseif meth == "wasser" #wasserstein is fixed. But which metric. That's next.
                    for metric in metric_index # iterate over metrics
                        if metric == 0 #coarse. the next question is, which lambda and epsilon to take?
                            for lambda in regularized_lambda_index #iterate over lambda indexes
                                for ep in epsilon_index
                                    iteration_hpc = iteration_hpc +  1 #increase the iteration
                                    if iteration_hpc == job_nr #********************
                                        d_to_take = copy(d)
                                        split_to_take = copy(split_nr)
                                        lr_wasser_to_use = meth
                                        metric_to_use = copy(metric)
                                        epsilon_to_use = copy(ep)
                                        lambda_to_use = copy(lambda)
                                    end
                                end
                            end
                        elseif metric == 1 #feature label. which kappa to take?
                            for kap in kappa_index #which lambda and epsilon to take
                                for lambda in regularized_lambda_index
                                    for ep in epsilon_index
                                        iteration_hpc = iteration_hpc +  1  #increase the iteration
                                        if iteration_hpc == job_nr #********************
                                            d_to_take = copy(d)
                                            split_to_take = copy(split_nr)
                                            lr_wasser_to_use = meth
                                            metric_to_use = copy(metric)
                                            epsilon_to_use = copy(ep)
                                            kappa_to_use = copy(kap)
                                            lambda_to_use = copy(lambda)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    ## READ DATASET, split tr/te
    local_path  = "/Users/.../Codes/"#update this
    dataset_names = vec(["breast-cancer","spect", "monks-3","tic-tac-toe", "kr-vs-kp","agaricus-lepiota", "lensesBIN", "balance-scaleBIN", "hayes-rothBIN", "lymphographyBIN", "carBIN", "splicesBIN", "balloons", "house-votes-84" ,"hiv", "nurseryBIN", "primacy-tumorBIN", "audiologyBIN"])
    dataset_names = local_path*"UCI/".*dataset_names
    split_nr = split_to_take #the split numbe.r.
    d = dataset_names[d_to_take]
    data = readdlm(d * "-cooked.csv", ',', Int64) #iterate over this
    Random.seed!(1234) #shuffle the rows -- always shuffle with the same structure
    data = data[Random.shuffle(1:end), :] #shuffle the rows ONCE
    n = size(data)[2] - 1 #last column is "y"
    #take raw data X_raw, y_raw out
    X_raw = data[:, 1:end-1]
    y_raw = data[:, end]
    #groups
    groups = vec(readdlm(d * "-cooked-key.csv", ',', String))[1:end-1] #don't take the last one
    groups = [eval(Meta.parse(replace(g, r"-" => ":"))) for g in groups]
    T = length(groups)#to use
    #random train/test split
    X, y, X_test, y_test = train_test_split(X_raw, y_raw, split_nr, total_split = 20)
    N, n = size(X)
    N_test, n_test = size(X_test)
    ## part 1 -- logistic regression
    lambdas = [0, (0.5)/(10^5),(0.5)/(10^4), (0.5)/(10^3),(0.5)/(10^2), (0.5)/10,  0.5] #lambda grid for CV
    if lr_wasser_to_use == "lr"
        lambda = lambdas[lambda_to_use]
        cv_errors = cv_logistic_regression(X,y, groups; regular = 1, lambda = lambda)
        avg_cv_error = sum(cv_errors)/5 #average CV error
        model, solver_time = logistic_regression(X,y, groups; regular = 1, lambda = lambda)
        optimal_obj_lr, beta_opt_lr, beta0_opt_lr = model_summarize(model) #save the solution
        error_lr = misclassification(X_test, y_test, beta_opt_lr, beta0_opt_lr)/N_test #test set error
        jldsave(local_path*"regData/regsave"*string(job_nr); solver_time, lambda, cv_errors, avg_cv_error, optimal_obj_lr, beta_opt_lr, beta0_opt_lr, groups, error_lr)
    end
    ## part 2 -- coarse Wasserstein
    lambdas = [0, (0.5)/(10^5), (0.5)/(10^3), (0.5)/10,  0.5] #lambda grid for DRO
    epsilons = [0, 1/(10^5), 1/(10^3), 1/10,  1] #epsilon grid for DRO
    kappas = [1 T]
    if lr_wasser_to_use == "wasser" && metric_to_use == 0
        lambda = lambdas[lambda_to_use]
        epsilon = epsilons[epsilon_to_use] #take the eps value
        cv_errors = cv_wasserstein(X, y, groups, epsilon; regular = 1, pen = lambda, dual_conic = 0, metric = 0)
        avg_cv_error = sum(cv_errors)/5 #average CV error of the epsilon we try
        #now we know the cv error of this parameter setting. Let us also fit to the training set, and save the answers.
        #fit the whole model
        model, iteration, solver_times = cutting_wasserstein(X, y, groups, epsilon; regular = 1, pen = lambda, dual_conic = 0, metric = 0) #or just cutting_wasserstein
        optimal_obj_coarse, beta_opt_coarse, beta0_opt_coarse = model_summarize(model)
        error_coarse = misclassification(X_test, y_test, beta_opt_coarse, beta0_opt_coarse)/N_test
        jldsave(local_path*"regData/regsave"*string(job_nr); epsilon, lambda, cv_errors, avg_cv_error, iteration, solver_times, optimal_obj_coarse, beta_opt_coarse, beta0_opt_coarse, error_coarse)
    end

    ## part 3 -- metric Wasserstein
    kappas = [1 T]
    if lr_wasser_to_use == "wasser" && metric_to_use == 1
        epsilon = epsilons[epsilon_to_use] #take the eps value
        lambda = lambdas[lambda_to_use]
        kappa = kappas[kappa_to_use]
        cv_errors = cv_wasserstein(X, y, groups, epsilon; regular = 1, pen = lambda, dual_conic = 0, metric = 1, p= 1, kappa=kappa)
        avg_cv_error = sum(cv_errors)/5 #average CV error of this (epsilon,lambda) combination
        #fit the whole model
        model, iteration, solver_times = cutting_wasserstein(X, y, groups, epsilon; regular = 1, pen=lambda, dual_conic = 0, metric = 1, p =1, kappa = kappa)
        optimal_obj_metric, beta_opt_metric, beta0_opt_metric = model_summarize(model)
        error_metric = misclassification(X_test, y_test, beta_opt_metric, beta0_opt_metric)/N_test
        jldsave(local_path*"regData/regsave"*string(job_nr); epsilon, lambda, kappa, cv_errors, avg_cv_error, iteration, solver_times, optimal_obj_metric, beta_opt_metric, beta0_opt_metric, error_metric)
    end
end
