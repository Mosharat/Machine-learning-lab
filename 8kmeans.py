import numpy as np

# K-means Clustering class
class KMeansClustering:
    def __init__(self, k):
        self.k = k

    def calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

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

    def train(self, data, initial_centroids):
        centroids = initial_centroids

        while True:
            clusters = self.assign_clusters(data, centroids)
            new_centroids = self.update_centroids(clusters)

            if np.array_equal(new_centroids, centroids):
                break

            centroids = new_centroids

        return clusters

# User interface for testing
def test_kmeans_clustering():
    # Data set
    data = np.array([
        [3.45],
        [3.78],
        [2.98],
        [3.24],
        [4.0],
        [3.9]
    ])

    # Initial centroids
    initial_centroids = np.array([
        [3.45],
        [2.98],
        [4.0]
    ])

    # Create a K-means clustering object with k=3
    kmeans = KMeansClustering(k=3)

    # Train the K-means clustering algorithm
    clusters = kmeans.train(data, initial_centroids)

    # Print the final clusters
    print("Final Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")

    # Test the K-means clustering with user input
    while True:
        cgpa = float(input("Enter the CGPA: "))

        point = np.array([cgpa])

        distances = [kmeans.calculate_distance(point, centroid) for centroid in initial_centroids]
        cluster_index = np.argmin(distances)

        print(f"The student belongs to Cluster {cluster_index + 1}")

        choice = input("Do you want to test another student? (y/n): ")
        if choice.lower() != 'y':
            break

# Run the K-means clustering algorithm
test_kmeans_clustering()
