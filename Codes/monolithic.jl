"A coarse metric that returns distance '1' if two instances are identical -- o.w. 0."
function coarse_metric(x_hat, y_hat, x, y)
    if (x_hat == x) && (y_hat == y)
        return 0.0
    end
    return 1.0
end
"The feature-label metric -- distance between features + kappa * label mismatch "
function feature_label_metric(x_hat, y_hat, x, y, p, kappa)
    return sum(x .!= x_hat)^(1/p) + (kappa * sum(y != y_hat))
end
"Feature label metric "
function generalized_feature_label_metric(x_hat, y_hat,groups, x, y, p, kappa)
    n = length(y_hat)
    T = length(groups)
    distance_to_return = (kappa * sum(y != y_hat))
    feature_distance = 0
    for i in groups
        feature_distance += !(x_hat[i] == x[i]) #check if the groups are the same
    end
    #or simply without for loop -> sum([x_hat[i] != x[i] for i in groups])
    distance_to_return = distance_to_return + (feature_distance)^(1/p)
    return distance_to_return/(kappa + T) #normalize the distance to stay in [0,1]
end
"Return the JuMP model of the monolithic Wasserstein DRO logistic regression."
function build_monolithic_model(X_train, y_train, groups, epsilon,regular,pen, dual, metric, p, kappa, restriction)
    N, n = size(X_train) #N rows, n predictors
    if dual == 0 #if we do not dualize
        model = Model() #start the model
        set_optimizer(model, MosekTools.Optimizer) #call MOSEK
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
    else
        model = Model(dual_optimizer(MosekTools.Optimizer)) #else, call MOSEK via dual_optimizer from the package Dualization
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
    end
    @variable(model, -100 <= beta[1:n] <= 100) #beta coefficients, set UB/LB arbitrary just to speed up calcs wlog.
    @variable(model, -100 <= beta0 <= 100) #intercept
    @variable(model, 0.0 <= s[1:N]) #auxiliary variables
    @variable(model, 0.0 <= lambda) #auxiliary variables
    if restriction == 1 #in case intercept = 0 is requested. This can be ignored. Was used for the stylized example in the paper's Section 2.2.
        @constraint(model, beta0 == 0)
    end
    #beta = 0 in features' last categories
    for g in groups
        if length(g) == 2
            error("a group length cannot be 2!")
        elseif length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #fix those coefficients to 0
        end
    end
    #define objective
    if regular == 2 #second order regularization
        @variable(model, 0.0 <= reg) #temp variable
        @constraint(model, [reg; [beta; beta0]] in SecondOrderCone()) #capture norm of betas
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N) + (pen * reg)) #penalize
    elseif regular == 1 #first order regularization
        @variable(model, 0.0 <= reg) #temp variable
        @constraint(model, [reg; [beta; beta0]] in MOI.NormOneCone(n + 2)) #capture norm of betas
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N) + (pen * reg)) #penalize
    else #means no regularization
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N))
    end
    iter = 0
    for y in [-1 1] #all possible targets
        for features in Iterators.product(collect(Iterators.repeated([-1, 1], n))...) #all possible features
            x = collect(features); #make an array
            u = -((x' * beta) + beta0) * y #will be used in softplus constraint
            for i in 1:N #iterate over the training set
                iter = iter + 1 #a new constraint is coming -- increase iteration
                #x_hat, y_hat will give us \xi_hat^i
                x_hat = X_train[i, :]
                y_hat = y_train[i]
                #now we take the distance below
                if metric == 0 #if metric is coarse
                    d = coarse_metric(x_hat, y_hat, x, y)
                else #else call the feature-label metric
                    d = generalized_feature_label_metric(x_hat, y_hat, groups, x, y, p, kappa)
                end
                #add a single conic constraint now
                softplus(model, s[i] + (lambda*d), u) #add the conic constraints
            end
        end
    end
    return model
end
"Build and optimize a JuMP model for monolithic Wasserstein DRO logistic regression."
function monolithic_wasserstein(X, y, groups, epsilon; regular = 0, pen = 0, dual_conic = 0, metric = 0, p = 1, kappa = 1, restriction =0)
    model = build_monolithic_model(X, y, groups, epsilon, regular, pen, dual_conic, metric, p, kappa, restriction)
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL #warn if not optimal
        error("Solution is not optimal.", termination_status(model))
    end
    solver_time = solve_time(model);
    #see solution
    #beta_opt = JuMP.value.(model[:beta])
    #optimal_obj = JuMP.objective_value(model)
    return model, solver_time
end
