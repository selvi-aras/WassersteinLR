"A given X_raw (features), y_raw (labels) pair is split into train:test"
function train_test_split(X_raw,y_raw, split; total_split = 20) #total_split means 20 possible splits
    N_raw, n = size(X_raw) #rows and nr predictors of the given data
    if split > total_split #is not possible
        error("split number is larger than the total allowed split!")
    end
    Random.seed!(split) #set the seetn according to the solut number
    test_instances = Random.shuffle(1:N_raw)[1:floor(Int, 0.2 * N_raw)]
    X = X_raw[Not(test_instances), :]
    y = y_raw[Not(test_instances)]
    X_test = X_raw[test_instances, :]
    y_test = y_raw[test_instances]
    return X, y, X_test, y_test
end
"Generate a dataset categorical features with N rows and n features"
function generate_dataset(N, n)
    true_beta = randn(n,1) # 'true' beta that we do not see
    true_beta0 = randn(1) # 'true' beta0
    normalization = norm([true_beta; true_beta0],2) #compute the norm
    true_beta = true_beta/normalization #normalize
    true_beta0 = true_beta0/normalization #normalize
    #training set below
    X_hat = rand([1 -1], N, n) #random -1/+1 data
    p_hat = 1 ./(1 .+ exp.(-X_hat*true_beta .- true_beta0)) #generate probabilities according to the true beta
    theta_hat = rand(N,1); #coin flips to sample ~ Bernuolli
    y_hat = (theta_hat .<= p_hat).* 2 .- 1 # ==1 means prob is less than cum.prob. so this becomes a "1" otherwise "-1"
    #test set has 100*N rows -- apply same steps for the test set
    X_test = rand([1 -1], 100*N, n)
    p_hat = 1 ./(1 .+ exp.(-X_test*true_beta .- true_beta0))
    theta_hat = rand(100*N,1)
    y_test = (theta_hat .<= p_hat).* 2 .- 1 # ==1 means prob is less than cum.prob. so this becomes a "1" otherwise "-1"
    return X_hat, y_hat, X_test, y_test
end

"Compute and return the number of misclassifications given a test set and beta values"
function misclassification(X_test, y_test, beta_opt, beta0_opt)
    p_hat_computed = 1 ./(1 .+ exp.(-X_test*beta_opt .- beta0_opt)); #classify according to the betas
    predictions_computed = (p_hat_computed .>= 0.5).*2 .-1; #+-1 vector of predictions
    mc = sum(predictions_computed .!= y_test)
    return mc
end
