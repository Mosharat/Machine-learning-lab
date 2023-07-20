import numpy as np

# Initialize weights and biases
weights = np.array([[-1, 1, -1], [-1, -1, 1]])
biases = np.array([3, 3])

# Define activation function
def activation(x):
    if x >= 0:
        return x
    else:
        return 0

def activationForward(x):
    return x

weigh2 = np.array([[1, -0.5], [-0.5, 1]])

# Define the forward pass
def forward(inputs, weights, biases):
    output = np.dot(weights, inputs) + biases
    return [activationForward(x) for x in output]
 
def recurrent(outputs, weigh2):
    output2 = np.dot(weigh2, outputs)
    return [activation(x) for x in output2]

# Define the fruit classification function
def classify_fruit(inputs):
    outputs = forward(inputs, weights, biases)
    outputs2 = recurrent(outputs, weigh2)
    if outputs2[0] > outputs2[1]:
        return "Banana"
    else:
        return "Pineapple"

# User input
test_input = input("Enter an array of integers: ")
test_input = test_input.split()
test_input = [int(x) for x in test_input]

# Test the operation of the network
print("Input:", test_input, "Output:", classify_fruit(test_input))
