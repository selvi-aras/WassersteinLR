"Return the 5-fold CV-errors of a fixed regularization setting in Logistic Regression (non-robust)."
function cv_logistic_regression(X,y, groups; regular = 0, lambda = 0)
    N, n = size(X) #size of the input
    break_points = round.(Int,LinRange(1,N,5+1)) #set of break-points, e.g., defines the "folds"
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] #collection of folds' indices
    errors = zeros(5) #will append
    for i = 1:5 #for i-th fold
        X_valid = X[vsets[i], :] #validation set is i-th fold
        y_valid = y[vsets[i]] #same for label vector
        N_valid, n_valid = size(X_valid) #size of the validation set
        X_train = X[Not(vsets[i]),:] #train set is all except for i-th fold
        y_train = y[Not(vsets[i])] #same for label vector
        model, solver_time = logistic_regression(X_train, y_train,groups, regular = regular, lambda = lambda) # solve the LR model with given parameters
        optimal_obj, beta_opt, beta0_opt = model_summarize(model) #get the optimal solution
        test_misclassification = misclassification(X_valid, y_valid, beta_opt, beta0_opt)/N_valid; #calculate the misclass. rate on the validation fold
        errors[i] = test_misclassification #push the i-th CVs error
    end
    return errors #return the 5-fold error vector
end

"Return the 5-fold CV-errors of a fixed parameter combination (regularization and ball radius) in Wasserstein Logistic Regression."
function cv_wasserstein(X, y, groups, epsilon; regular= 0, pen= 0, dual_conic = 0, metric = 0, p = 1, kappa = 1)
    N, n = size(X) #size of the given set
    break_points = round.(Int,LinRange(1,N,5+1)) #set of break-points
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] #each fold indices
    errors = zeros(5)
    for i = 1:5 #5-fold CV
        X_valid = X[vsets[i], :] #validation set is i-th fold
        y_valid = y[vsets[i]] #same
        N_valid, n_valid = size(X_valid)
        X_train = X[Not(vsets[i]),:] #train set is all except for i-th fold
        y_train = y[Not(vsets[i])]
        model, iteration, solver_time = cutting_wasserstein(X_train, y_train, groups, epsilon; regular = regular, pen = pen, dual_conic = dual_conic, metric = metric, p = p, kappa = kappa);
        optimal_obj, beta_opt, beta0_opt = model_summarize(model)
        test_misclassification = misclassification(X_valid, y_valid, beta_opt, beta0_opt)/N_valid;
        errors[i] = test_misclassification #push the i-th CVs error
    end
    return errors
end
