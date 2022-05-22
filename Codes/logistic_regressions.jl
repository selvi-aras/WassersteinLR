"Takes a model and adds a softplus constraint. See https://jump.dev/JuMP.jl/stable/tutorials/conic/logistic_regression"
function softplus(model, t, linear_transform) #exponential cone constraint
    # models constraints of form:
    # log(1 + exp(- linear_transform)) <= t
    # will be called from logistic regression building
    z = @variable(model, [1:2], lower_bound = 0.0)
    #add the exp-cone constraints
    @constraint(model, sum(z) <= 1.0)
    @constraint(model, [linear_transform - t, 1, z[1]] in MOI.ExponentialCone())
    @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
end

"Same as softplus function, but returns the variables corresponding to the added constraints (to delete some constraints later)."
function softplus_updated(model, t, linear_transform)
    z = @variable(model, [1:2], lower_bound = 0.0)
    cn1 = @constraint(model, sum(z) <= 1.0)
    cn2 = @constraint(model, [linear_transform - t, 1, z[1]] in MOI.ExponentialCone())
    cn3 = @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
    return z,cn1,cn2,cn3
end
"Returns a JuMP model to optimize for logistic regression."
function build_logit_model(X, y, groups, regular, lambda)
    N, n = size(X) #N rows, n binary predictors
    model = Model(dual_optimizer(MosekTools.Optimizer)) #start the model via MOSEK
    set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
    @variable(model, beta[1:n]) #beta coefficients
    @variable(model, beta0) #intercept
    @variable(model, t[1:N]) #auxiliary variables
    for i in 1:N #add N softplus constraints, e.g., log-loss at i-th point <= t_i
        u = -(X[i, :]' * beta + beta0) * y[i]
        softplus(model, t[i], u) 
    end
    #now make the last betas = 0
    for g in groups
        if length(g) == 2
            error("a group length cannot be 2!")
        elseif length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #fix those coefficients to 0 to inactivate the last cateogry of a one-hot encoded feature.
            #the reason we keep all categories in one-hot encoding and enforcing one beta = 0 instead of dropping one column directly is that the Wasserstein
            #ball would be impacted from asymmetry, e.g., 0-0 vs 1-0 have a distance of 1, while 1-0 vs 0-1 have a distance of 2.
        end
    end
    # Define objective, which depends on whether we take regularization
    if regular == 2 #second order regularization
        @variable(model, 0.0 <= reg)
        @constraint(model, [reg; [beta; beta0]] in SecondOrderCone())
        @objective(model, Min, sum(t)/N + (lambda * reg))
    elseif regular == 1 #first order regularization
        @variable(model, 0.0 <= reg)
        @constraint(model, [reg; [beta; beta0]] in MOI.NormOneCone(n + 2))
        @objective(model, Min, sum(t)/N + (lambda * reg))
    else #no regularization
        @objective(model, Min, sum(t)/N)
    end
    return model
end
"Take data and return optimized logistic regression model."
function logistic_regression(X,y, groups; regular = 0, lambda = 0)
    # Optimizes the logistic regression problem
    model = build_logit_model(X, y, groups, regular, lambda)
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL #warn if not optimal
        error("Solution is not optimal.")
    end
    solver_time = solve_time(model)
    #see solution
    #beta_opt = JuMP.value.(model[:beta])
    #optimal_obj = JuMP.objective_value(model)
    return model, solver_time #question: does returning the model make things slow?
end

"For a given (optimized) JuMP model, return optimal decisions and the variables."
function model_summarize(model)
    optimal_obj = JuMP.objective_value(model);
    beta_opt = JuMP.value.(model[:beta])
    beta0_opt =  JuMP.value.(model[:beta0])
    #time = solve_time(model);
    return optimal_obj, beta_opt, beta0_opt
end
