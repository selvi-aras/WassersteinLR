function mixed_cv_logistic_regression(X_cont,X,y, groups; regular = 0, lambda = 0) #same with logistic regression function but returns cv'ed error
    N, n = size(X)
    n_cont = size(X_cont)[2]
    break_points = round.(Int,LinRange(1,N,5+1)) #set of break-points
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] #each fold indices
    errors = zeros(5)
    for i = 1:5 #5-fold CV
        X_cont_valid = X_cont[vsets[i], :] #validation set is i-th fold
        X_valid = X[vsets[i], :] #validation set is i-th fold
        y_valid = y[vsets[i]] #same

        N_valid, n_valid = size(X_valid)
        n_cont_valid = size(X_cont_valid)[2]

        X_cont_train = X_cont[Not(vsets[i]),:] #train set is all except for i-th fold
        X_train = X[Not(vsets[i]),:] #train set is all except for i-th fold
        y_train = y[Not(vsets[i])]
        model, solver_time = mixed_logistic_regression(X_cont_train, X_train, y_train,groups, regular = regular, lambda = lambda);
        optimal_obj, beta_cont_opt, beta_opt, beta0_opt = mixed_model_summarize(model)
        test_misclassification = mixed_misclassification(X_cont_valid, X_valid, y_valid, beta_cont_opt, beta_opt, beta0_opt)/N_valid;
        errors[i] = test_misclassification #push the i-th CVs error
    end
    return errors
end

function mixed_cv_wasserstein(X_cont, X, y, groups, epsilon; regular= 0, pen= 0, dual_conic = 0, metric = 0, p = 1, kappa = 1) #same with logistic regression function but returns cv'ed error
    N, n = size(X)
    n_cont = size(X_cont)[2]
    break_points = round.(Int,LinRange(1,N,5+1)) #set of break-points
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] #each fold indices
    errors = zeros(5)
    for i = 1:5 #5-fold CV
        X_cont_valid = X_cont[vsets[i], :] #validation set is i-th fold
        X_valid = X[vsets[i], :] #validation set is i-th fold
        y_valid = y[vsets[i]] #same

        N_valid, n_valid = size(X_valid)
        n_cont_valid = size(X_cont_valid)[2]

        X_cont_train = X_cont[Not(vsets[i]),:] #train set is all except for i-th fold
        X_train = X[Not(vsets[i]),:] #train set is all except for i-th fold
        y_train = y[Not(vsets[i])]
        model, iteration, solver_time = mixed_cutting_wasserstein(X_cont_train, X_train, y_train, groups, epsilon; regular = regular, pen = pen, dual_conic = dual_conic, metric = metric, p = p, kappa = kappa);
        optimal_obj, beta_cont_opt, beta_opt, beta0_opt = mixed_model_summarize(model)
        test_misclassification = mixed_misclassification(X_cont_valid, X_valid, y_valid, beta_cont_opt, beta_opt, beta0_opt)/N_valid;
        errors[i] = test_misclassification #push the i-th CVs error
    end
    return errors
end
