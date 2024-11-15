# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(all(X[, 1] == 1) && all(Xt[, 1] == 1)){
  } else {
    stop("First column of X and Xt are not ones:Please check!")
  }
  # Check for compatibility of dimensions between X and Y
  if(nrow(X) == length(y)){
  } else {
    stop("Dimension of X and Y are not compatible,PLease check!")
  }
  # Check for compatibility of dimensions between Xt and Yt
  if(nrow(Xt) == length(yt)){
  } else {
    stop("Dimension of Xt and yt are not compatible,PLease check!")
  }
  # Check for compatibility of dimensions between X and Xt
  if(ncol(X) == ncol(Xt)){
  } else {
    stop("Dimension of X and Xt are not compatible,PLease check!")
  }
  # Check eta is positive
  if(eta >0){
  } else {
    stop("Eta is not positive:Check!")
  }
  # Check lambda is non-negative
  if(lambda >= 0){
  } else {
    stop("Lambda is negative:Check!")
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  n<-nrow(X)
  p<-ncol(X)
  K<-length(unique(y))
  
  if(is.null(beta_init)){
    beta_init<-matrix(0,nrow = p,ncol = K)
  } else {
    if(nrow(beta_init)!=p || ncol(beta_init)!=K){
      stop("dimensions of beta_initial are not suitable.Please check!")
    }
  }
  #Initialsation of train/test/object value arrays
  error_train<-array(0,dim=numIter + 1)
  error_test<-array(0,dim=numIter + 1)
  objective<-array(0,dim=numIter + 1)
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  error_train[1]<-objective_fun(beta_init,X,y,lambda)$error
  error_test[1]<-objective_fun(beta_init,Xt,yt,lambda)$error
  objective[1]<-objective_fun(beta_init,X,y,lambda)$objective_value
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
 
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}