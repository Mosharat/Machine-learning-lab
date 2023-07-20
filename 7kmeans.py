import numpy as np

# K-means Clustering class
class KMeansClustering:
    def __init__(self, k):
        self.k = k

    def initialize_centroids(self, data):
        np.random.shuffle(data)
        return data[:self.k]

    def calculate_distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    def assign_clusters(self, data, centroids):
        clusters = [[] for _ in range(self.k)]

        for point in data:
            distances = [self.calculate_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        return clusters

    def update_centroids(self, clusters):
        centroids = []
        for cluster in clusters:
            centroids.append(np.mean(cluster, axis=0))
        return np.array(centroids)

    def train(self, data):
        centroids = self.initialize_centroids(data)

        while True:
            clusters = self.assign_clusters(data, centroids)
            new_centroids = self.update_centroids(clusters)

            if np.array_equal(new_centroids, centroids):
                break

            centroids = new_centroids

        return clusters, centroids

# User interface for testing
def test_kmeans_clustering():
    # Data set
    data = np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [3.0, 4.0],
        [5.0, 7.0],
        [3.5, 5.0],
        [4.5, 5.0],
        [3.5, 4.5]
    ])

    # Create a K-means clustering object with k=2
    kmeans = KMeansClustering(k=2)

    # Train the K-means clustering algorithm
    clusters, centroids = kmeans.train(data)

    # Print the clusters and centroids
    print("Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")

    print("\nCentroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i+1}: {centroid}")

    # Test the K-means clustering with user input
    while True:
        point_a = float(input("Enter the value for variable A: "))
        point_b = float(input("Enter the value for variable B: "))

        point = np.array([point_a, point_b])

        distances = [kmeans.calculate_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)

        print(f"The point belongs to Cluster {cluster_index + 1}")

        choice = input("Do you want to test another point? (y/n): ")
        if choice.lower() != 'y':
            break

# Run the K-means clustering algorithm
test_kmeans_clustering()
