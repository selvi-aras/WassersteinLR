function mixed_model_summarize(model)
    optimal_obj = JuMP.objective_value(model)
    beta_cont_opt = JuMP.value.(model[:beta_cont])
    beta_opt = JuMP.value.(model[:beta])
    beta0_opt =  JuMP.value.(model[:beta0])
    return optimal_obj, beta_cont_opt, beta_opt, beta0_opt
end

function mixed_build_logit_model(X_cont, X, y, groups, regular, lambda)
    # will build the model to optimize for a logistic regression
    N, n = size(X) #N rows, n predictors
    n_cont = size(X_cont)[2]
    model = Model() #start the model
    rel_gap = 10^-7
    set_optimizer(model, MosekTools.Optimizer)
    set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
    set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
    set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    @variable(model, beta_cont[1:n_cont]) #beta coefficients
    @variable(model, beta[1:n]) #beta coefficients
    @variable(model, beta0) #intercept
    @variable(model, t[1:N]) #auxiliary variables
    for i in 1:N
        u = -(X_cont[i, :]' * beta_cont + X[i, :]' * beta + beta0) * y[i]
        softplus(model, t[i], u) 
    end
    #now make the last betas = 0
    for g in groups
        if length(g) == 2
            error("a group length cannot be 2!")
        elseif length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #fix those coefficients to 0
        end #else, we have length(g) = 1 which means
    end
    # Define objective, which depends on whether we take regularization
    if regular == 2 #second order regularization
        @variable(model, 0.0 <= reg)
        @constraint(model, [reg; [beta_cont; beta; beta0]] in SecondOrderCone())
        @objective(model, Min, sum(t)/N + (lambda * reg))
    elseif regular == 1 #first order regularization
        @variable(model, 0.0 <= reg)
        @constraint(model, [reg; [beta_cont; beta; beta0]] in MOI.NormOneCone(n + n_cont + 2))
        @objective(model, Min, sum(t)/N + (lambda * reg))
    else #means no regularization
        @objective(model, Min, sum(t)/N)
    end

    return model
end

function mixed_logistic_regression(X_cont, X,y, groups; regular = 0, lambda = 0)
    # Optimizes the mixed-feature logistic regression problem
    model = mixed_build_logit_model(X_cont, X, y, groups, regular, lambda)
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL #warn if not optimal
        error("Solution is not optimal.")
    end
    solver_time = solve_time(model);
    #see solution
    #beta_opt = JuMP.value.(model[:beta])
    #optimal_obj = JuMP.objective_value(model)
    return model, solver_time
end
