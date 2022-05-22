#this file reads HPC output and makes meaning out of them.
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
local_path  = "/Users/.../Codes/"#update this

dataset_names = local_path*"UCI/".*dataset_names

lambdas = [0, (0.5)/(10^5),(0.5)/(10^4), (0.5)/(10^3),(0.5)/(10^2), (0.5)/10,  0.5] #lambda grid
lambdas_simplified = [0, (0.5)/(10^5), (0.5)/(10^3), (0.5)/10,  0.5] #DRO lambda grid
epsilons = [0, 1/(10^5), 1/(10^3), 1/10,  1]

nr_overall = max_split*(length(lambdas) + length(lambdas_simplified)*length(epsilons) + 2*length(lambdas_simplified)*length(epsilons))

d_to_take = 1 #give the dataset index
data = readdlm(dataset_names[d_to_take] * "-cooked.csv", ',', Int64) #iterate over thi
N_raw, n_raw = size(data)
#start
errors_lr = zeros(max_split) #test error of the logistic regression
errors_coarse = zeros(max_split) #test error of the best coarse DRO
errors_metric_1 = zeros(max_split) #test error of the best metric DRO wiht kappa = 1
errors_metric_T = zeros(max_split) #test error of the best metric DRO wiht kappa = T
#best epsilons to save
best_epsilons_coarse = zeros(max_split) #best epsilon according to the CV error
best_epsilons_metric_1 = zeros(max_split)
best_epsilons_metric_T = zeros(max_split)
#best lambdas to save
best_lambdas_lr = zeros(max_split)
best_lambdas_coarse = zeros(max_split) #best epsilon according to the CV error
best_lambdas_metric_1 = zeros(max_split)
best_lambdas_metric_T = zeros(max_split)
for split_to_take in 1:max_split
    running_count = (d_to_take - 1)*nr_overall + (split_to_take - 1)*(length(lambdas) + 3*length(lambdas_simplified)*length(epsilons)) #start the counter from here
    #test errorss
    error_lr = 0 #test error of the logistic regression
    error_coarse = 0 #test error of the best coarse DRO
    error_metric_1 = 0 #test error of the best metric DRO wiht kappa = 1
    error_metric_T = 0 #test error of the best metric DRO wiht kappa = T
    #best parameters
    best_epsilon_coarse = 0 #best epsilon according to the CV error
    best_epsilon_metric_1 = 0
    best_epsilon_metric_T = 0
    best_lambda_lr = 0
    best_lambda_coarse = 0
    best_lambda_metric_1 = 0
    best_lambda_metric_T = 0

    # START the main reading algorithm, i.e., for a fixed dataset and split, take performances. Start with LR
    #part 1 - LR
    best_cv_error = 1
    for lambda in lambdas
        running_count += 1
        obj = load("./regData/regsave"*string(running_count))
        if obj["avg_cv_error"] <= best_cv_error
            best_cv_error = obj["avg_cv_error"]
            best_lambda_lr = lambda
            error_lr = obj["error_lr"]
        end
    end
    # now do the coarse ones (not reported in the paper)
    best_cv_error = 1
    for lambda in lambdas_simplified
        for epsilon in epsilons
            running_count += 1
            obj = load("./regData/regsave"*string(running_count))
            if obj["avg_cv_error"] <= best_cv_error
                best_cv_error = obj["avg_cv_error"]
                best_epsilon_coarse = epsilon
                best_lambda_coarse = lambda
                error_coarse = obj["error_coarse"]
            end
        end
    end
    # now do the metric-1 one
    best_cv_error = 1
    for lambda in lambdas_simplified
        for epsilon in epsilons
            running_count += 1
            obj = load("./regData/regsave"*string(running_count))
            if obj["avg_cv_error"] <= best_cv_error
                best_cv_error = obj["avg_cv_error"]
                best_epsilon_metric_1 = epsilon
                best_lambda_metric_1  = lambda
                error_metric_1 = obj["error_metric"]
            end
        end
    end

    # now do the metric-T one
    best_cv_error = 1
    for lambda in lambdas_simplified
        for epsilon in epsilons
            running_count += 1
            obj = load("./regData/regsave"*string(running_count))
            if obj["avg_cv_error"] <= best_cv_error
                best_cv_error = obj["avg_cv_error"]
                best_epsilon_metric_T = epsilon
                best_lambda_metric_T = lambda
                error_metric_T = obj["error_metric"]
            end
        end
    end
    # save them all
    errors_lr[split_to_take], errors_coarse[split_to_take],errors_metric_1[split_to_take], errors_metric_T[split_to_take] =
        error_lr, error_coarse, error_metric_1, error_metric_T
    best_epsilons_coarse[split_to_take],best_epsilons_metric_1[split_to_take], best_epsilons_metric_T[split_to_take] =
        best_epsilon_coarse, best_epsilon_metric_1, best_epsilon_metric_T
    best_lambdas_lr[split_to_take], best_lambdas_coarse[split_to_take],best_lambdas_metric_1[split_to_take], best_lambdas_metric_T[split_to_take] =
        best_lambda_lr, best_lambda_coarse, best_lambda_metric_1, best_lambda_metric_T
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
quantiles = [quantile!(errors_lr, 0.9),quantile!(errors_coarse, 0.9), quantile!(errors_metric_1, 0.9), quantile!(errors_metric_T, 0.9)]
println(round.(medians[[1,3,4]], digits = 4).*100)
