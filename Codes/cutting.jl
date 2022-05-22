"In case metric = 0 (coarse metric) is taken, this finds the most violated constraint."
function most_violated_coarse(X, y, val_beta, val_beta0, val_s, val_lambda)
    max_viol = -0.1 # to be appended
    if val_beta0 >= 0 #if intercept is non-negative
        sol1_x = sign.(val_beta); #sol1_x is the sign vector
        sol1_y = -1; #sol1_y is -1
    else
        sol1_x = -sign.(val_beta); #else otjher way arpound
        sol1_y = 1;
    end
    loc = argmin(val_s); #otherwise take min of s_i over i \in [N] as the constraints are identical accross all i
    violation = log(1 + exp(-sol1_y * (sol1_x'*val_beta + val_beta0))) - val_lambda - val_s[loc];
    if violation > max_viol
        max_viol = violation
        adversarial_x = sol1_x; #worst-case
        adversarial_y = sol1_y; #worst-case
        adversarial_d = 1;
        adversarial_loc = loc; #this is the one defining s
    end
    #oterwise consider each data-point and mimic these
    violations_vector = log.(1 .+ exp.(- y .* (X*val_beta .+ val_beta0))) .- val_s
    loc = argmax(vec(violations_vector));
    if violations_vector[loc] > max_viol
        max_viol = violations_vector[loc]
        adversarial_x = X[loc,:]; #worst-case
        adversarial_y = y[loc];
        adversarial_d = 0;
        adversarial_loc = loc;
    end
    return max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc
end
"Find the most violated constarint for the ground metric being feature-label metric."
function most_violated_feature_label_metric(X, y, groups, p,kappa,val_beta, val_beta0, val_s, val_lambda)
    max_viol = -0.1 # to be appended
    N, n = size(X)
    T = length(groups) #number of groups
    singletons = [k[1] for k in groups if length(k) == 1] #singleton groups
    #initiate the solutions to return
    adversarial_x = copy(X[1, :]); #worst-case
    adversarial_y = 1; #worst-case
    adversarial_d = 1.0;
    adversarial_loc = 1; #this is the one defining s
    if groups == [kk:kk for kk in 1:n] #means the groups are all singleton so no one-hot encoding -- slightly easier implementation
        for i = 1:N #iterate over xi_hat
            x_hat = X[i, :] #i-th instance's predictors
            y_hat = y[i] #i-th instance's target
            for case in [1 -1] #we fix the "y" as either y_hat or - y_hat (no other options possible)
                sol1_y = y_hat*case #take sol1_y (candidate adversarial y)
                to_sort = (-1 * sol1_y) .* (vec(val_beta) .* vec(x_hat))
                sorted_indices = sortperm(to_sort, rev = true) #sort once in the beginning, keep the indexes. Will be used later.
                for k in 0:n #number of times x and x_hat agree
                    sol1_d = (1/(n + kappa))*((n-k)^(1/p) + kappa*(case == -1))#candidate for d. Now this is fixed. case = -1 means y and y_hat are different
                    if k == n
                        sol1_x = copy(x_hat) #always copy
                    elseif k == 0
                        sol1_x = -1*x_hat #always opposite as k = 0
                    else
                        top_k_indices = sorted_indices[1:k] #take the k largest indices.
                        sol1_x = -1*x_hat #copy the training predictors
                        sol1_x[top_k_indices] = -1 * sol1_x[top_k_indices] #switch the sign of the selected indexes
                    end
                    violation = log(1 + exp(-sol1_y * (sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
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
                    sol1_d = (1/(T + kappa))*((T-k)^(1/p) + kappa*(case == -1)) #candidate for d. Now this is fixed. case = -1 means y and y_hat are different
                    if k == T #if T components agree, then nothing to do
                        sol1_x = copy(x_hat) #always copy
                    elseif k == 0 #if 0 components agree, just disagree everywhere greedily
                        #this means that all the groups will disagree
                        #next, greedily pick HOW to disagree
                        sgn_vec = copy(sign_vec_to_use)
                        sgn_vec[singletons] .= -1 #however singletons need to be -1 no matter what -- make the singletons disagree
                        sol1_x = sgn_vec.*x_hat #copy x_hat but disagree on components where sgn_vec = -1
                    else #otherwise, agree in k groups greedily by looking at "sorted_groups"
                        sgn_vec = copy(sign_vec_to_use) #normally this is fine
                        sgn_vec[singletons] .= -1 #make the singletons disagree. OK, now everything disagrees.
                        #now, sgn_vec consists of all disagreements. But we need to agree the top k groups!
                        #make them agree next
                        flip = vcat([group_indexes for group_indexes in groups[sorted_groups[1:k]]]...) #collect the agreeing indexes
                        sgn_vec[flip] .= 1
                        sol1_x = sgn_vec.*x_hat
                    end
                    #violation at the current design
                    violation = log(1 + exp(-sol1_y * (sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
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
"Solve the Wasserstein DRO Logistic Regression problem via the proposed cutting-plane method."
function cutting_wasserstein_updated(X, y, groups, epsilon; regular = 0, pen = 0 , dual_conic = 0, metric = 0, p = 1, kappa = 1, restriction = 0)
    ################# Parameters are the following:
    # X -> input matrix; y -> target vector; epsilon -> Wasserstein radius
    #regular -> 0: no regul., 1: ell_1 regul., 2: ell_2 regul.
    #pen -> regularization penalty
    #dual_conic -> 0: solve primal, 1: solve dual conic optimization problem.
    #metric -> 0: coarse metric, 1: feature-label metric
    #p -> (feature-label metric) norm of the feature difference in W-ball
    #kappa -> (feature-label metric) weight of the target difference in W-ball
    #restriction = 1 imposes the intercept to be forced to 0. We used this for stylized ex in S2.2.
    ##################
    # create a basic model below
    if epsilon == 0 #epsilon = 0 means no robustness, call the logistic regression model
        model, solver_time = logistic_regression(X,y, groups; regular = regular, lambda = pen, restriction = restriction)
        iteration = 1 #no cutting plane iterations
        return model, iteration, solver_time
    end
    # otherwise, epsilon > 0, and we have
    solver_times = zeros(0)
    N, n = size(X) #N rows, n predictors
    rel_gap = 10^-7 #to prevent numerical issues -- what is the feasibility threshold.
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
    @variable(model, -100 <= beta[1:n] <= 100) #beta coefficients -- UB LB are arbitrary
    @variable(model, -100 <= beta0 <= 100) #intercept
    @variable(model, 0.0 <= s[1:N]) #auxiliary variables
    @variable(model, 0.0 <= lambda) #auxiliary variables
    if restriction == 1 #just for stylized ex
        @constraint(model, beta0 == 0)
    end
    #now add the beta = 0 constraints for dummy groups
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
        @constraint(model, [reg; [beta; beta0]] in SecondOrderCone()) #capture norm of betas
        @objective(model, Min, (lambda*epsilon) + (sum(s)/N) + (pen * reg)) #penalize
    elseif regular == 1 #first order regularization
        @variable(model, 0.0 <= reg) #temp variable
        @constraint(model, [reg; [beta; beta0]] in MOI.NormOneCone(n + 2)) #capture norm of betas
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
    vars, c1s, c2s, c3s = vec([]),vec([]),vec([]),vec([]) #to append -- constraints and variables added to the model
    base_to_take = 100 #every this many iterations we delete the most slacked constraints.
    slack_to_take = 0.99 #slack to delete (between 0-1 -> a value closer to zero is a larger violation, hence the more this value is the more constraints we delete)
    while violated == 1 #while there is still some violation in the problem
        iteration = iteration + 1 #keep track of the cutting plane iterations
        #now find the most violated constraint (does not depend on regularization)
        if metric == 0
            max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc = most_violated_coarse(X, y, JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]))
        else
            max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc = most_violated_feature_label_metric(X, y,groups, p,kappa,JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]))
        end
        if max_viol > rel_gap #if violation is still non-zero
            #add the constraint
            var, c1, c2, c3 = softplus_updated(model, s[adversarial_loc] + (lambda*adversarial_d), -((adversarial_x' * beta) + beta0) * adversarial_y) #add the most violated constraint
            #done with the constraints
            #add the latest constraints
            push!(vars, var)
            push!(c1s, c1)
            push!(c2s, c2)
            push!(c3s, c3)
            #optimize
            JuMP.optimize!(model) #re-optimize
            push!(solver_times, solve_time(model)) #add the solver time
            #every base_to_take-th iteration delete some slack'ed constraints
            if iteration == Int(base_to_take)
                base_to_take = round(Int64, 1.5*base_to_take) #delete constarints less often, keep updating -- prevents possible loops
                indexes_to_del = vec([]) #constarints to delete
                for (c_ind, c_to_del) in enumerate(c1s) #go over every constraint we have
                    if value(c_to_del) <= slack_to_take #if slack, delete
                        push!(indexes_to_del, c_ind)
                    end
                end
                slack_to_take = max(0.0, slack_to_take - 0.01) #keep reducing the threshold so we are less strict -- prevents possible loops
                #delete the constraints
                for ind in indexes_to_del
                    delete(model, vars[ind])
                    delete(model, c1s[ind])
                    delete(model, c2s[ind])
                    delete(model, c3s[ind])
                end
                #update the vector of all constraints and remove the deleted ones
                deleteat!(vars, indexes_to_del)
                deleteat!(c1s, indexes_to_del)
                deleteat!(c2s, indexes_to_del)
                deleteat!(c3s, indexes_to_del)
                iteration = iteration +  1 #new iter
                JuMP.optimize!(model) #optimize for the first time
                push!(solver_times, solve_time(model))
            end
        else
            violated = 0 #no more violations
        end
    end
    if termination_status(model) != OPTIMAL #warn if not optimal
        print("Solution is not optimal with error code ", termination_status(model) )
    end
    #optimal_obj = JuMP.objective_value(model)
    return model, iteration, solver_times #question: does returning the model make things slow?
end
