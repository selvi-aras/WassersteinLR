"identical with generate_data.jl, however, this is for the case where we have mixed features."
function mixed_train_test_split(X_cont_raw, X_raw,y_raw, split; total_split = 20)
    N_raw, n = size(X_raw)
    n_cont = size(X_cont_raw)[2] #number of continuous variables
    if split > total_split
        error("split number is larger than the total allowed split!")
    end
    Random.seed!(split) #set the seed
    test_instances = Random.shuffle(1:N_raw)[1:floor(Int, 0.2 * N_raw)]
    #take out training
    X_cont = X_cont_raw[Not(test_instances), :]
    X = X_raw[Not(test_instances), :]
    y = y_raw[Not(test_instances)]
    #take out test
    X_cont_test = X_cont_raw[test_instances, :]
    X_test = X_raw[test_instances, :]
    y_test = y_raw[test_instances]
    return X_cont, X, y, X_cont_test, X_test, y_test
end

"Generate a dataset categorical features with N rows, n binary features, and n_cont continuous features."
function mixed_generate_dataset(N, n, n_cont)
    true_beta_cont = randn(n_cont,1); # 'true' beta for continuous variables
    true_beta = randn(n,1); # 'true' beta assuming there is a truth
    true_beta0 = randn(1);  # 'true' beta0
    #normalize the betas
    normalization = norm([true_beta_cont; true_beta; true_beta0],2); #from the linear algebra package
    true_beta_cont = true_beta_cont/normalization;
    true_beta = true_beta/normalization;
    true_beta0 = true_beta0/normalization;
    #construct training set
    X_cont_hat = rand(N,n_cont) #continuous features are [0,1]
    X_hat = rand([1 -1], N, n)
    p_hat = 1 ./(1 .+ exp.(-(X_cont_hat*true_beta_cont) - (X_hat*true_beta) .- true_beta0)) #generate probabilities
    theta_hat = rand(N,1) #coin flips to sample
    y_hat = (theta_hat .<= p_hat).* 2 .- 1; # ==1 means prob is less than cum.prob. so this becomes a "1" otherwise "-1"
    #construct test set
    inflation = 100
    X_cont_test = rand(inflation*N, n_cont)
    X_test = rand([1 -1], inflation*N, n); #10 times more test instances
    p_hat = 1 ./(1 .+ exp.(-X_cont_test*true_beta_cont -X_test*true_beta .- true_beta0)); #generate probabilities
    theta_hat = rand(inflation*N,1); #coin flips to sample
    y_test = (theta_hat .<= p_hat).* 2 .- 1; # ==1 means prob is less than cum.prob. so this becomes a "1" otherwise "-1"
    return X_cont_hat, X_hat, y_hat, X_cont_test, X_test, y_test
end

"Returns the msc.rate of model in a test set"
function mixed_misclassification(X_cont_test, X_test, y_test, beta_cont_opt, beta_opt, beta0_opt)
    p_hat_computed = 1 ./(1 .+ exp.(-X_cont_test*beta_cont_opt -X_test*beta_opt .- beta0_opt)); #generate probabilities
    predictions_computed = (p_hat_computed .>= 0.5).*2 .-1; #+-1 vector of predictions
    mc = sum(predictions_computed .!= y_test)
    return mc
end
