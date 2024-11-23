library(ggplot2)

# Initialize the parameters
set.seed(50)   #Seed fixing
n_train <- 500  # Number of training samples
n_val <- 200    # Number of testing samples
p <- 10         # Number of input features
K <- 3          # Number of classes
hidden_p <- 20  # Size of hidden layer
scale <- 1e-3   # Scale for weight initialization
lambda <- 0.01  # Regularization parameter
rate <- 0.01    # Learning rate
mbatch <- 50    # Batch size for SGD
nEpoch <- 50    # Number of epochs

# training data
X_train <- matrix(rnorm(n_train * p), nrow = n_train, ncol = p)
y_train <- sample(0:(K - 1), n_train, replace = TRUE)

# Testing data
X_val <- matrix(rnorm(n_val * p), nrow = n_val, ncol = p)
y_val <- sample(0:(K - 1), n_val, replace = TRUE)

# Training neural network
result <- NN_train(X_train, y_train, X_val, y_val, 
                   lambda = lambda, rate = rate, 
                   mbatch = mbatch, nEpoch = nEpoch, 
                   hidden_p = hidden_p, scale = scale)

#training and testing errors
train_error <- result$error
val_error <- result$error_val

# Plot the training and testing error over epochs
df <- data.frame(Epoch = 1:nEpoch, 
                 TrainingError = train_error, 
                 ValidationError = val_error)

# Plot using ggplot2
ggplot(df, aes(x = Epoch)) +
  geom_line(aes(y = TrainingError, color = "Training Error")) +
  geom_line(aes(y = ValidationError, color = "Testing Error")) +
  labs(title = "Training and Testing Error Over Epochs",
       y = "Error (%)", x = "Epoch",
       color = "Error Type") +
  theme_minimal()

# Final validation accuracy
final_test_error <- tail(val_error, 1)
cat("Final Validation Error: ", final_test_error, "%\n")

#Changing the parameters
result1 <- NN_train(X_train, y_train, X_val, y_val, 
                   lambda = 0.03, rate = 0.02, 
                   mbatch = 100, nEpoch = 50, 
                   hidden_p = hidden_p, scale = scale)

train_error <- result1$error
val_error <- result$error_val

# Plot the training and validation error over epochs
df <- data.frame(Epoch = 1:nEpoch, 
                 TrainingError = train_error, 
                 ValidationError = val_error)

# Plot using ggplot2
ggplot(df, aes(x = Epoch)) +
  geom_line(aes(y = TrainingError, color = "Training Error")) +
  geom_line(aes(y = ValidationError, color = "Testing Error")) +
  labs(title = "Training and Testing Error Over Epochs",
       y = "Error (%)", x = "Epoch",
       color = "Error Type") +
  theme_minimal()

# Final validation accuracy
final_test_error <- tail(val_error, 1)
cat("Final Validation Error: ", final_test_error, "%\n")

