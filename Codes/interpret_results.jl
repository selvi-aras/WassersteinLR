#this file reads HPC output and compiles all the results
using LinearAlgebra
using JLD2
using PlotlyJS
using DataFrames
using DelimitedFiles
using Statistics
#meta parameters
dataset_index = 1:18
max_split = 20 #20 tr/te splits
dataset_names = vec(["breast-cancer","spect", "monks-3","tic-tac-toe", "kr-vs-kp", "agaricus-lepiota", "lensesBIN", "balance-scaleBIN", "hayes-rothBIN", "lymphographyBIN", "carBIN", "splicesBIN", "balloons", "house-votes-84" ,"hiv", "nurseryBIN", "primacy-tumorBIN", "audiologyBIN"])
epsilons = [0,1/100000, 1/10000, 1/1000, 1/100, 1/10, 1]
local_path  = "/Users/.../Codes/"#update this
dataset_names = local_path*"UCI/".*dataset_names
nr_overall = max_split*(1 + length(epsilons) + 2*length(epsilons)) #to reserve lengths

d_to_take = 1 # use the first dataset for now
data = readdlm(dataset_names[d_to_take] * "-cooked.csv", ',', Int64) #iterate over this
N_raw, n_raw = size(data)
#start
errors_lr = zeros(max_split) #test error of the logistic regression
errors_coarse = zeros(max_split) #test error of the best coarse DRO
errors_metric_1 = zeros(max_split) #test error of the best metric DRO wiht kappa = 1
errors_metric_T = zeros(max_split) #test error of the best metric DRO wiht kappa = T
best_epsilons_coarse = zeros(max_split) #best epsilon according to the CV error
best_epsilons_metric_1 = zeros(max_split)
best_epsilons_metric_T = zeros(max_split)
for split_to_take in 1:max_split
    running_count = (d_to_take - 1)*nr_overall + (split_to_take - 1)*(1 + length(epsilons) + 2*length(epsilons)) #start the counter from here
    #test errorss
    error_lr = 0 #test error of the logistic regression
    error_coarse = 0 #test error of the best coarse DRO
    error_metric_1 = 0 #test error of the best metric DRO wiht kappa = 1
    error_metric_T = 0 #test error of the best metric DRO wiht kappa = T
    #best parameters
    best_epsilon_coarse = 0 #best epsilon according to the CV error
    best_epsilon_metric_1 = 0
    best_epsilon_metric_T = 0
    # START the main reading algorithm, i.e., for a fixed dataset and split, take performances. Start with LR
    running_count += 1
    obj = load("./Data/save"*string(running_count))
    error_lr = obj["error_lr"]
    # now do the coarse ones
    best_cv_error = 1
    for epsilon in epsilons
        running_count += 1
        obj = load("./Data/save"*string(running_count))
        if obj["avg_cv_error"] <= best_cv_error
            best_cv_error = obj["avg_cv_error"]
            best_epsilon_coarse = epsilon
            error_coarse = obj["error_coarse"]
        end
    end
    # now do the metric-1 one
    best_cv_error = 1
    for epsilon in epsilons
        running_count += 1
        obj = load("./Data/save"*string(running_count))
        if obj["avg_cv_error"] <= best_cv_error
            best_cv_error = obj["avg_cv_error"]
            best_epsilon_metric_1 = epsilon
            error_metric_1 = obj["error_metric"]
        end
    end
    # now do the metric-T one
    best_cv_error = 1
    for epsilon in epsilons
        running_count += 1
        obj = load("./Data/save"*string(running_count))
        if obj["avg_cv_error"] <= best_cv_error
            best_cv_error = obj["avg_cv_error"]
            best_epsilon_metric_T = epsilon
            error_metric_T = obj["error_metric"]
        end
    end
    # save them all
    errors_lr[split_to_take], errors_coarse[split_to_take],errors_metric_1[split_to_take], errors_metric_T[split_to_take] =
        error_lr, error_coarse, error_metric_1, error_metric_T
    best_epsilons_coarse[split_to_take],best_epsilons_metric_1[split_to_take], best_epsilons_metric_T[split_to_take] =
        best_epsilon_coarse, best_epsilon_metric_1, best_epsilon_metric_T
end
#size
println("N: ", N_raw, " n: ", n_raw)
#summarize errors
df_errors = DataFrame(hcat(errors_lr, errors_coarse,errors_metric_1, errors_metric_T), ["LR", "Coarse", "Metric1", "MetricT"])
df_errors = stack(df_errors)
#some stats
means = [Statistics.mean(errors_lr), Statistics.mean(errors_coarse), Statistics.mean(errors_metric_1), Statistics.mean(errors_metric_T)]
medians = [median!(errors_lr), median!(errors_coarse), median!(errors_metric_1), median!(errors_metric_T)]
sds = sqrt.([var(errors_lr), var(errors_coarse), var(errors_metric_1), var(errors_metric_T)])
quantiles = [quantile!(errors_lr, 0.95),quantile!(errors_coarse, 0.95), quantile!(errors_metric_1, 0.95), quantile!(errors_metric_T, 0.95)]
println(round.(medians[[1,3,4]], digits = 4).*100) #report the medians, omit the coarse metric as in the paper for space considerations we only use feature-label metric
