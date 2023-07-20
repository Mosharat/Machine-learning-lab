#Here's a Python program that implements the perceptron learning rule for the given training set:
import numpy as np

# define the training set
X = np.array([[1, 0, 1], [0, -1, -1], [-1, -0.5, -1]])
y = np.array([-1, 1, 1])

# initialize the weight vector
w = np.array([1, -1, 0])

# set the learning rate and maximum number of iterations
alpha = 0.1
max_iterations = 100

# train the neural network using the perceptron learning rule
for i in range(max_iterations):
    for j in range(len(X)):
        x = X[j]
        y_hat = np.sign(np.dot(w, x))
        error = y[j] - y_hat
        w =w+ alpha * error * x

    # check for convergence
    if np.all(np.sign(np.dot(X, w)) == y):
        print(f"Converged after {i+1} iterations.")
        break
    else:
        print(f"Did not converge after {max_iterations} iterations.")

print("Final weight vector: ", w)

# test the network with new input patterns
x_test = np.array([[1, 1, 1], [-1, 0, 1], [0, 1, -1]])
y_test = np.sign(np.dot(x_test, w))
print("Predictions for test set: ", y_test)
