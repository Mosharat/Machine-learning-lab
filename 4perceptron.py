import numpy as np
import matplotlib.pyplot as plt

# Define the input vectors and their desired output values
p1 = np.array([2, 2])
t1 = 0
p2 = np.array([1, -2])
t2 = 1
p3 = np.array([-2, 2])
t3 = 0
p4 = np.array([-1, 1])
t4 = 1

# Initialize the weight vector and bias
W = np.array([0, 0])
b = 0

# Define the activation function
def activation(x):
    return 1 if x >= 0 else 0

# Train the neural network using the perceptron rule
max_iterations = 1000
for i in range(max_iterations):
    # Apply each input vector in order
    for p, t in [(p1, t1), (p2, t2), (p3, t3), (p4, t4)]:
        # Calculate the weighted sum of inputs
        a = np.dot(W, p) + b
        # Calculate the output of the neuron
        y = activation(a)
        # Update the weights and bias if the output is incorrect
        if y != t:
            W = W + (t - y) * p
            b = b + (t - y)

    # Check if the problem is solved
    if (np.dot(W, p1) + b >= 0 and np.dot(W, p3) + b < 0) or \
       (np.dot(W, p1) + b < 0 and np.dot(W, p3) + b >= 0) or \
       (np.dot(W, p2) + b >= 0 and np.dot(W, p4) + b < 0) or \
       (np.dot(W, p2) + b < 0 and np.dot(W, p4) + b >= 0):
        break

# Print the solution
print("The problem is solved after", i+1, "iterations.")
print("The weight vector is", W)
print("The bias is", b)

# Plot the input vectors and the decision boundary
plt.scatter(p1[0], p1[1], c='g', marker='o')
plt.scatter(p2[0], p2[1], c='b', marker='o')
plt.scatter(p3[0], p3[1], c='r', marker='s')
plt.scatter(p4[0], p4[1], c='black', marker='s')
x = np.linspace(-3, 3)
y = (-b - W[0]*x) / W[1]
plt.plot(x, y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Classification')
plt.show()
