# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0,K)
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed)
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p) # Input to hidden layer weights
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K) # Hidden to output layer weights
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  # Extracting dimension
  n <- length(y)
  
  # changing the class label to 1:K
  y <- y + 1
  
  # Calculating P_K - n x k matrix with each row p_k(x_i), k = 0, ..., K-1
  mat_exp <- exp(scores)
  P_K <- mat_exp/ rowSums(mat_exp)
  
  # Creating a matrix with I(Y == k) as rows
  Ymat <- 0 + outer(y, 1:K, function(a, b) a == b)
  
  # [ToDo] Calculate loss when lambda = 0
  loss <- -sum(Ymat * log(P_K)) / n
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # error = ...
  error <- 100 * (sum(y != max.col(P_K, ties.method = "first")) / n)
  #pred_labels <- apply(scores, 1, which.max) - 1  # Subtract 1 for 0-based labels
  #error <- mean(pred_labels != y) * 100
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad = ...
  grad <- (-Ymat + P_K) / n
  
  #exp_scores <- exp(scores)
  #prob <- exp_scores / rowSums(exp_scores)
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  # Extracting Dimensions
  n <- length(y)
  h <- length(b1)
  p <- ncol(X)
  # [To Do] Forward pass
  # From input to hidden 
  A1 <- X %*% W1 + matrix(b1, n, h, byrow = T)
  # ReLU
  H <- pmax(A1, 0)
  # From hidden to output scores
  scores <- H %*% W2 + matrix(b2, n, K, byrow = T)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, scores, K)
  out_grad <- out$grad # n x K matrix
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- crossprod(H, out_grad) + lambda * W2 # h x K matrix
  db2 <- colSums(out_grad) # vector of length K
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH <- tcrossprod(out_grad, W2) # n x h matrix
  dA1 <- dH * (A1 > 0) # I(A>0) # n x h matrix
  dW1 <- crossprod(X, dA1) + lambda * W1 # p x h matrix
  db1 <- colSums(dA1) # vector of length h
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)
  
  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}