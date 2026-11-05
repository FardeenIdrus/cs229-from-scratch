import numpy as np


import numpy as np

np.random.seed(0)  # same random numbers every run

m = 100  # number of training examples

features = np.random.randn(m, 2)  # 100 rows x 2 cols of random numbers: x1, x2
ones = np.ones((m, 1))            # 100 rows x 1 col of 1s: x0, the intercept feature
X = np.hstack([ones, features])   # join side by side: 100 x 3, each row is [1, x1, x2]

true_weights = np.array([4, 3, -2])  # [theta0, theta1, theta2], the answer we hope to recover
noise = 0.5 * np.random.randn(m)     # small random error for each example

# y = θᵀx + ε 
# Create the targets y. We chose the weights, so we know the right answer.
# X @ true_weights = true prediction per example: 1*4 + x1*3 + x2*(-2)
y = X @ true_weights + noise 


print(X.shape, y.shape)  # should print (100, 3) (100,)

# Use the normal equation to directly compute the vector of weights that
# minimise the cost function. This is done by taking the partial derivative 
# of the cost function with respect to each weight and setting it to 0
# allowing us to obtain the weight values at the global minimum of the cost function

# Normal Equation : θ = (XᵀX)⁻¹Xᵀy 
thetha = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(thetha)

# Gradient descent
# Initialise the weights to 0 initially 
thetha1 = np.zeros(3)
# Set the learning rate
learning_rate = 0.1
# 1000 iterations 
for i in range(1000):
    # Compute the model's prediction at each iteration as 
    # (row of feature values for each example * weight for each feature)
    # Repeated for all examples (100)
     h = X @ thetha1
     # Compute error as (y-h)
     error = y-h
     # Slope  = (y-h)x j,i 
    # for each weight j: error x feature j, summed over all examples, divided by m
     slope = X.T @ error /m 
     # Update rule 
     thetha1 = thetha1 + learning_rate * slope
    
print(thetha1)
     
     
     