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
function hpc(job_nr)
    ##now iterate over all the parameter combinations
    #************JOB NR MUST BE GIVEN
    #job_nr = 1
    #
    dataset_index = 1:18 #we have 18 datasets
    max_split = 20 #20 tr/te splits
    lr_wasser = ["lr", "wasser"] #method logistic or wasserstein
    metric_index = [0 1] #0 is coarse, 1 is the feature-label one. In the paper we only report feature-label metric.
    epsilon_index = 1:7 #7 epsilons will be CV'ed
    kappa_index = 1:2 #two kappas will be tried kappa = 1 (e.g., as if label was another feature), or kappa = T (e.g., as if label is equally important with all features)
    iteration_hpc = 0 #running counter

    # things to return
    d_to_take = 1 #dataset index to use later
    split_to_take = 1 #dataset split to use later
    lr_wasser_to_use = "lr"
    metric_to_use = 0
    epsilon_to_use = 1
    kappa_to_use  = 1

    #given a job_nr map it to the parameter setting we will use (for HPC.)
    for d in dataset_index #iterate over dataset
        for split_nr in 1:max_split #iterate over splits
            for meth in lr_wasser #iterate over all methods
                if meth == "lr"
                    iteration_hpc = iteration_hpc +  1 #A Method is being called
                    if iteration_hpc == job_nr #******************** capture the setting we need
                        d_to_take = copy(d)
                        split_to_take = copy(split_nr)
                        lr_wasser_to_use = meth
                    end
                elseif meth == "wasser" #wasserstein is fixed. But which metric. That's next.
                    for metric in metric_index # iterate over metrics
                        if metric == 0 #coarse. Question is, which epsilon to take?
                            for ep in epsilon_index
                                iteration_hpc = iteration_hpc +  1 #increase the iteration
                                if iteration_hpc == job_nr #******************* capture the setting we need*
                                    d_to_take = copy(d)
                                    split_to_take = copy(split_nr)
                                    lr_wasser_to_use = meth
                                    metric_to_use = copy(metric)
                                    epsilon_to_use = copy(ep)
                                end
                            end
                        elseif metric == 1 #feature label. which kappa to take?
                            for kap in kappa_index #which epsilon to take
                                for ep in epsilon_index
                                    iteration_hpc = iteration_hpc +  1  #increase the iteration
                                    if iteration_hpc == job_nr #******************** capture the setting we need
                                        d_to_take = copy(d)
                                        split_to_take = copy(split_nr)
                                        lr_wasser_to_use = meth
                                        metric_to_use = copy(metric)
                                        epsilon_to_use = copy(ep)
                                        kappa_to_use = copy(kap)
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
    local_path  = "/Users/.../"#update this
    dataset_names = vec(["breast-cancer","spect", "monks-3","tic-tac-toe", "kr-vs-kp", "agaricus-lepiota", "lensesBIN", "balance-scaleBIN", "hayes-rothBIN", "lymphographyBIN", "carBIN", "splicesBIN", "balloons", "house-votes-84" ,"hiv", "nurseryBIN", "primacy-tumorBIN", "audiologyBIN"])
    dataset_names = local_path*"UCI/".*dataset_names #assuming all datasets are in a folder named UCI
    split_nr = split_to_take #the split number.
    d = dataset_names[d_to_take] #dataset name is extacted
    data = readdlm(d * "-cooked.csv", ',', Int64) #iterate over this
    Random.seed!(1234) #shuffle the rows -- always shuffle with the same structure to reproduce results easily
    data = data[Random.shuffle(1:end), :] #shuffle rows once
    n = size(data)[2] - 1 #last column is "y"
    #take raw data X_raw, y_raw out
    X_raw = data[:, 1:end-1]
    y_raw = data[:, end]
    #groups
    groups = vec(readdlm(d * "-cooked-key.csv", ',', String))[1:end-1] #don't take the last one as that is the label's column
    groups = [eval(Meta.parse(replace(g, r"-" => ":"))) for g in groups]
    T = length(groups)#to use
    #random train/test split
    X, y, X_test, y_test = train_test_split(X_raw, y_raw, split_nr, total_split = 20)
    N, n = size(X)
    N_test, n_test = size(X_test)
    ## part 1 -- logistic regression -- nothign to CV!
    if lr_wasser_to_use == "lr"
        model, solver_time = logistic_regression(X,y,groups, regular = 0) #fit the full model
        optimal_obj_lr, beta_opt_lr, beta0_opt_lr = model_summarize(model) #save the solution
        error_lr = misclassification(X_test, y_test, beta_opt_lr, beta0_opt_lr)/N_test #test set error
        jldsave(local_path*"Data/save"*string(job_nr); solver_time, optimal_obj_lr, beta_opt_lr, beta0_opt_lr, groups, error_lr)
    end
    ## part 2 -- coarse Wasserstein
    epsilons = [0, 1/100000, 1/10000, 1/1000, 1/100, 1/10, 1]
    kappas = [1 T]
    if lr_wasser_to_use == "wasser" && metric_to_use == 0
        epsilon = epsilons[epsilon_to_use] #take the eps value
        cv_errors = cv_wasserstein(X, y, groups, epsilon; metric = 0)
        avg_cv_error = sum(cv_errors)/5 #average CV error of the epsilon we try
        #now we know the cv error of this epsilon. Let us also fit to the training set, and save the answers.
        #fit the whole model
        model, iteration, solver_times = cutting_wasserstein(X, y, groups, epsilon; metric = 0) #or just cutting_wasserstein
        optimal_obj_coarse, beta_opt_coarse, beta0_opt_coarse = model_summarize(model)
        error_coarse = misclassification(X_test, y_test, beta_opt_coarse, beta0_opt_coarse)/N_test
        #IMPORTANT: we train a model on the whole dataset as well, however, we decide which model to use according to the average cv-error. The model that is trained on the full dataset is only used after we decide which model to keep.
        jldsave(local_path*"Data/save"*string(job_nr); epsilon, cv_errors, avg_cv_error, iteration, solver_times, optimal_obj_coarse, beta_opt_coarse, beta0_opt_coarse, error_coarse)
    end

    ## part 3 -- metric Wasserstein
    if lr_wasser_to_use == "wasser" && metric_to_use == 1
        epsilon = epsilons[epsilon_to_use] #take the eps value
        kappa = kappas[kappa_to_use]
        cv_errors = cv_wasserstein(X, y, groups, epsilon; regular = 0, pen = 0, dual_conic = 1, metric = 1, p= 1, kappa=kappa)
        avg_cv_error = sum(cv_errors)/5 #average CV error of the epsilon we try
        #now we know the cv error of this epsilon. Let us also fit to the training set, and save the answers.
        #fit the whole model
        model, iteration, solver_times = cutting_wasserstein(X, y, groups, epsilon; regular = 0, pen=0, dual_conic = 1, metric = 1, p =1, kappa = kappa)
        optimal_obj_metric, beta_opt_metric, beta0_opt_metric = model_summarize(model)
        error_metric = misclassification(X_test, y_test, beta_opt_metric, beta0_opt_metric)/N_test
        jldsave(local_path*"Data/save"*string(job_nr); epsilon, kappa, cv_errors, avg_cv_error, iteration, solver_times, optimal_obj_metric, beta_opt_metric, beta0_opt_metric, error_metric)
    end
end
