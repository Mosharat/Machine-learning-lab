import math
from collections import Counter

# Define the data points
data = [
    (158, 58, "M"), (158, 59, "M"), (158, 63, "M"),
    (160, 59, "M"), (160, 60, "M"), (163, 60, "M"),
    (163, 61, "M"), (160, 64, "L"), (163, 64, "L"),
    (165, 61, "L"), (165, 62, "L"), (165, 65, "L"),
    (168, 62, "L"), (168, 63, "L"), (168, 66, "L"),
    (170, 63, "L"), (170, 64, "L"), (170, 68, "L")
]

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Function to predict T-shirt size using KNN algorithm
def predict_tshirt_size(data, height, weight, k):
    distances = []
    for point in data:
        point_height, point_weight, size = point
        distance = euclidean_distance((height, weight), (point_height, point_weight))
        distances.append((distance, size))
    
    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[0])
    
    # Select the top k nearest neighbors
    nearest_neighbors = distances[:k]
    
    # Count the occurrences of each T-shirt size in the nearest neighbors
    size_counts = Counter(neighbor[1] for neighbor in nearest_neighbors)
    
    # Get the most common T-shirt size
    predicted_size = size_counts.most_common(1)[0][0]
    
    return predicted_size

# User interface
def main():
    height = int(input("Enter Mr. Perfect's height (in cm): "))
    weight = int(input("Enter Mr. Perfect's weight (in kg): "))
    k = int(input("Enter the value of k (3 or 5): "))
    
    predicted_size = predict_tshirt_size(data, height, weight, k)
    
    print("Predicted T-shirt size for Mr. Perfect: ", predicted_size)

# Run the program
main()

