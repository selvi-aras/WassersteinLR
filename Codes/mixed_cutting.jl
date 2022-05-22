function mixed_most_violated_coarse(X_cont, X, y, val_beta_cont, val_beta, val_beta0, val_s, val_lambda)
    max_viol = -0.1 # to be appended
    #eventually we will return those, but below we override them with worst-cases
    adversarial_x = copy(X[1, :])#worst-case
    adversarial_y = 1 #worst-case y
    adversarial_d = 1.0 #worst-case distance metric
    adversarial_loc = 1 #worst-case i index
    N, n = size(X)
    for i = 1:N #iterate over the training points
        #Case 1 - all identical
        sol1_x = copy(X[i, :]) #just the same as the X we see
        sol1_y = copy(y[i]) #identical
        violation = log(1 + exp(-sol1_y * ((X_cont[i, :]'*val_beta_cont) + (sol1_x'*val_beta) + val_beta0))) - val_s[i] #d = 0 in  this case
        if violation > max_viol
            adversarial_x = sol1_x
            adversarial_y = sol1_y
            adversarial_d = 0
            adversarial_loc = i
        end
        #Case 2 - they disagree, design greedily
        sol1_x = -sign.(val_beta) #just the same as the X we see
        sol1_y = 1 #sol1_y is -1
        violation = log(1 + exp(-sol1_y * ((X_cont[i, :]'*val_beta_cont) + (sol1_x'*val_beta) + val_beta0))) - val_lambda - val_s[i]
        if violation > max_viol
            adversarial_x = sol1_x
            adversarial_y = sol1_y
            adversarial_d = 1
            adversarial_loc = i
        end
        #same but y = -1
        sol1_x = sign.(val_beta) #just the same as the X we see
        sol1_y = -1 #sol1_y is -1
        violation = log(1 + exp(-sol1_y * ((X_cont[i, :]'*val_beta_cont) + (sol1_x'*val_beta) + val_beta0))) - val_lambda - val_s[i]
        if violation > max_viol
            adversarial_x = sol1_x
            adversarial_y = sol1_y
            adversarial_d = 1
            adversarial_loc = i
        end
    end
    return max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc
end

function mixed_most_violated_feature_label_metric(X_cont, X, y, groups, p,kappa, val_beta_cont, val_beta, val_beta0, val_s, val_lambda)
    max_viol = -0.1 # to be appended
    N, n = size(X) #maybe "n" comes from cutting_wasserstein? do we need to redefine?
    n_cont = size(X_cont)[2]
    T = length(groups) #number of groups
    singletons = [k[1] for k in groups if length(k) == 1] #singleton groups of categorical variables
    adversarial_x = copy(X[1, :]); #worst-case
    adversarial_y = 1; #worst-case
    adversarial_d = 1.0;
    adversarial_loc = 1; #this is the one defining s (location of s[\hat{\xi}]), or, the "worst-case i index"
    if groups == [kk:kk for kk in 1:n] #means the groups are all singleton so no one-hot encoding!
        for i = 1:N #iterate over xi_hat
            x_hat = X[i, :] #i-th instance's predictors
            y_hat = y[i] #i-th instance's target
            for case in [1 -1] #we fix the "y" as either y_hat or - y_hat (no other options possible)
                sol1_y = y_hat*case #take sol1_y (candidate adversarial y)
                to_sort = (-1 * sol1_y) .* (vec(val_beta) .* vec(x_hat))
                sorted_indices = sortperm(to_sort, rev = true) #sort once in the beginning, keep the indexes. Will be used later.
                for k in 0:n #number of times x and x_hat agree
                    sol1_d = (1/(n + n_cont + kappa))*((n-k)^(1/p) + kappa*(case == -1))#candidate for d. Now this is fixed. case = -1 means y and y_hat are different
                    if k == n
                        sol1_x = copy(x_hat) #always copy
                    elseif k == 0
                        sol1_x = -1*x_hat #always opposite as k = 0
                    else
                        top_k_indices = sorted_indices[1:k] #take the k largest indices.
                        sol1_x = -1*x_hat #copy the training predictors
                        sol1_x[top_k_indices] = -1 * sol1_x[top_k_indices] #switch the sign of the selected indexes
                    end
                    violation = log(1 + exp(-sol1_y * (X_cont[i, :]'*val_beta_cont + sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
                    if violation > max_viol
                        max_viol = copy(violation)
                        adversarial_x = copy(sol1_x); #worst-case
                        adversarial_y = copy(sol1_y); #worst-case
                        adversarial_d = copy(sol1_d);
                        adversarial_loc = copy(i); #this is the one defining s
                    end
                end
            end
        end
    else
        for i = 1:N #iterate over xi_hat
            x_hat = X[i, :] #i-th instance's predictors
            y_hat = y[i] #i-th instance's target
            for case in [1 -1] #we fix the "y" as either y_hat or - y_hat (no other options possible)
                sol1_y = y_hat*case #take sol1_y (candidate adversarial y)
                #Now the Agree and Disagree vectors
                values_to_check = - sol1_y .* vec(val_beta) .* vec(x_hat)
                A = [sum(values_to_check[g]) for g in groups] #sum by group
                values_D = max.(0, values_to_check)
                D = [length(g) > 1 ? sum(values_D[g]) : 0  for g in groups] #0 if the group is of length 1
                utilities = vec(A .- D)
                sign_vec_to_use = sign.(vec(values_to_check)) #this is if we had greedily chosen everything, will be used later
                #
                sorted_groups = sortperm(utilities, rev = true) #sort once in the beginning, keep the indexes. Will be used later.
                for k in 0:T #number of groups that agree between x and x_hat
                    sol1_d = (1/(T + n_cont + kappa))*((T-k)^(1/p) + kappa*(case == -1)) #candidate for d. Now this is fixed. case = -1 means y and y_hat are different
                    if k == T
                        sol1_x = copy(x_hat) #always copy
                    elseif k == 0
                        #this means that all the groups will disagree
                        sgn_vec = copy(sign_vec_to_use)
                        sgn_vec[singletons] .= -1 #however singletons need to be -1 no matter what -- make the singletons disagree
                        sol1_x = sgn_vec.*x_hat #always opposite as k = 0
                    else
                        sgn_vec = copy(sign_vec_to_use) #normally this is fine
                        sgn_vec[singletons] .= -1 #make the singletons disagree. OK, now everything disagrees.
                        #make them agree next
                        flip = vcat([group_indexes for group_indexes in groups[sorted_groups[1:k]]]...) #collect the agreeing indexes
                        sgn_vec[flip] .= 1
                        sol1_x = sgn_vec.*x_hat
                    end
                    violation = log(1 + exp(-sol1_y * (X_cont[i, :]'*val_beta_cont + sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
                    if violation > max_viol
                        max_viol = copy(violation)
                        adversarial_x = copy(sol1_x); #worst-case
                        adversarial_y = copy(sol1_y); #worst-case
                        adversarial_d = copy(sol1_d);
                        adversarial_loc = copy(i); #this is the one defining s
                    end
                end
            end
        end
    end
    #fill above
    return max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc
end

function mixed_cutting_wasserstein(X_cont, X, y, groups, epsilon; regular = 0, pen = 0 , dual_conic = 0, metric = 0, p = 1, kappa = 1)
    ################# Parameters are the same as the function cutting_wasserstein_updated in "cutting.jl"
    # however, we only have an extra X_cont
    ##################
    # create a basic model below
    if epsilon == 0 #then we simply have the LR!
        model, solver_time = mixed_logistic_regression(X_cont, X,y, groups; regular = regular, lambda = pen)
        iteration = 1
        return model, iteration, solver_time
    end
    solver_times = zeros(0)
    N, n = size(X) #N rows, n predictors
    n_cont = size(X_cont)[2]
    T = length(groups)
    rel_gap = 10^-7
    if dual_conic == 0 #dualize or not decide
        model = Model() #start the model
        set_optimizer(model, MosekTools.Optimizer)
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    else
        model = Model(dual_optimizer(MosekTools.Optimizer)) #start the model
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1, "MSK_IPAR_INTPNT_MULTI_THREAD" => 0) #one thread
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    end
    #add the variables
    @variable(model, -100 <= beta_cont[1:n_cont] <= 100) #beta_cont coefficients
    @variable(model, -100 <= beta[1:n] <= 100) #beta coefficients
    @variable(model, -100 <= beta0 <= 100) #intercept
    @variable(model, 0.0 <= s[1:N]) #auxiliary variables
    @variable(model, 0.0 <= lambda) #auxiliary variables
    #dual norm constraints
    @constraint(model, beta_cont[1:n_cont] .<= lambda/(T + n_cont + kappa)) #dual norm constraints (divide by T + n_cont + kappa to make sure metric in [0,1] always)
    @constraint(model, -lambda/(T + n_cont + kappa) .<= beta_cont[1:n_cont] ) #dual norm constraints
    #now add the beta 0 constarints
    for g in groups
        if length(g) == 2
            error("a group length cannot be 2!")
        elseif length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #fix those coefficients to 0
        end #else, we have length(g) = 1 which means
    end
    #now define the objective function regarding whether we take regularization
    if regular == 2 #second order regularization
        @variable(model, 0.0 <= reg) #temp variable
        @constraint(model, [reg; [beta_cont; beta; beta0]] in SecondOrderCone()) #capture norm of betas
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N) + (pen * reg)) #penalize
    elseif regular == 1 #first order regularization
        @variable(model, 0.0 <= reg) #temp variable
        @constraint(model, [reg; [beta_cont; beta; beta0]] in MOI.NormOneCone(n + 1 + n_cont + 1)) #capture norm of betas
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N) + (pen * reg)) #penalize
    else #means no regularization
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N))
    end
    #optimize once first
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL #warn if not optimal
        error("Solution is not optimal.")
    end
    push!(solver_times, solve_time(model))
    #start the cutting-plane
    iteration = 0;
    violated = 1;
    #new elements
    vars, c1s, c2s, c3s = vec([]),vec([]),vec([]),vec([]) #to append -- variables etc
    added_constraints = Array{Vector{Float64},1}() #empty vector of vectors
    deleted_constraints = Array{Vector{Float64},1}() #index of deleted constraints
    #
    while violated == 1 #while there is still some violation in the problem
        iteration = iteration + 1 #keep track of the cutting plane iterations
        #now find the most violated constraint (does not depend on regularization)
        if metric == 0
            max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc = mixed_most_violated_coarse(X_cont, X, y, JuMP.value.(model[:beta_cont]), JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]))
        else
            max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc = mixed_most_violated_feature_label_metric(X_cont, X, y, groups, p,kappa, JuMP.value.(model[:beta_cont]), JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]))
        end
        if max_viol > rel_gap
            #add the constraint
            var, c1, c2, c3 = softplus_updated(model, s[adversarial_loc] + (lambda*adversarial_d), -((X_cont[adversarial_loc,:]' * beta_cont) + (adversarial_x' * beta) + beta0) * adversarial_y) #add the most violated constraint
            push!(added_constraints, vcat([adversarial_loc], [adversarial_d], adversarial_x, [adversarial_y]))
            #done with the constraints
            #add the latest constraints
            push!(vars, var)
            push!(c1s, c1)
            push!(c2s, c2)
            push!(c3s, c3)
            #optimize
            JuMP.optimize!(model) #optimize for the first time
            push!(solver_times, solve_time(model))
            #every 200-th iteration delete constraints with more than "0.1" slack
            if iteration % 200 == 0
                indexes_to_del = vec([])
                for (c_ind, c_to_del) in enumerate(c1s)
                    if value(c_to_del) <= 0.9
                        if !(added_constraints[c_ind] in deleted_constraints)
                            push!(indexes_to_del, c_ind)
                            push!(deleted_constraints, added_constraints[c_ind])
                        end
                    end
                end
                #delete the constraints
                for ind in indexes_to_del
                    delete(model, vars[ind])
                    delete(model, c1s[ind])
                    delete(model, c2s[ind])
                    delete(model, c3s[ind])
                end
                #update the vector of all constarints as they are deleted now
                deleteat!(vars, indexes_to_del)
                deleteat!(c1s, indexes_to_del)
                deleteat!(c2s, indexes_to_del)
                deleteat!(c3s, indexes_to_del)
                deleteat!(added_constraints, indexes_to_del)
                #new iter
                iteration = iteration +  1
                JuMP.optimize!(model) #optimize for the first time
                push!(solver_times, solve_time(model))
            end
        else
            violated = 0 #no more violations
        end
    end
    if termination_status(model) != OPTIMAL #warn if not optimal
        error("Solution is not optimal with error code ", termination_status(model) )
    end
    #optimal_obj = JuMP.objective_value(model)
    return model, iteration, solver_times
end
