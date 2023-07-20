import numpy as np

# Perceptron class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        if summation >= 0:
            return "Orange"
        else:
            return "Apple"

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += label * inputs
                self.bias += label

# User interface for testing
def test_perceptron():
    # Training data
    training_inputs = np.array([
        [1, -1, -1],  # Orange
        [1, 1, -1]    # Apple
    ])

    labels = np.array([1, -1])  # 1 for Orange, -1 for Apple

    # Create a perceptron with three input dimensions
    perceptron = Perceptron(3)

    # Train the perceptron
    perceptron.train(training_inputs, labels, epochs=10)

    # Test the perceptron with user input
    while True:
        shape = float(input("Enter the shape (-1 for elliptical, 1 for round): "))
        texture = float(input("Enter the texture (-1 for rough, 1 for smooth): "))
        weight = float(input("Enter the weight (-1 for less than one pound, 1 for more than one pound): "))

        fruit = np.array([weight, texture, shape])
        prediction = perceptron.predict(fruit)

        print("The fruit is:", prediction)

        choice = input("Do you want to test another fruit? (y/n): ")
        if choice.lower() != 'y':
            break

# Run the perceptron network
test_perceptron()
