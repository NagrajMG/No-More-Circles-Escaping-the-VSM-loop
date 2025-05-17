import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Clustering():
    def __init__(self):
        self.centroids = None
        self.doc_labels = None
        self.k = 22
        self.docs = np.load("Main_project_code/fast_search/output/originalDocs.npy")
    
    def cluster(self):
        """Perform KMeans clustering on the document vectors."""
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.doc_labels = kmeans.fit_predict(self.docs)
        self.centroids = kmeans.cluster_centers_
        
        # Save the cluster labels and centroids
        np.save("Main_project_code/fast_search/output/cluster_labels.npy", self.doc_labels)
        np.save("Main_project_code/fast_search/output/centroids.npy", self.centroids)

    def elbowPlot(self, max_k= 50):
        distortions = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.docs)
            distortions.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, distortions, 'bo-', linewidth=2)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Distortion)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.savefig("Main_project_code/fast_search/output/elbow_plot_for_approximated.png")
        plt.close()


if __name__ == "__main__":
    clustering = Clustering()
    # clustering.elbowPlot(max_k=50)  
    clustering.cluster()
    print("Clustering completed and results saved.")
