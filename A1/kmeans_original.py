import numpy as np

class KmeansModel:
    def __init__(self, X, k, max_iters):
        self.X = X
        self.k = k
        self.max_iters = max_iters
        self.dim = X.shape[1]
        self.N = X.shape[0]
        
        self.centroids = np.zeros((self.k, self.dim))
        initial_labels = np.zeros((self.N))
        indices = np.random.choice(self.N, self.k, replace=False)
        self.centroids = self.X[indices]
        
        initial_labels = self.get_labels(self.X, self.centroids)
    
    def get_labels(self, X, centroids):
        labels = []
        for i, point in enumerate(X):
            distance = np.linalg.norm(point - centroids, axis=1)
            labels.append(np.argmin(distance))
        return np.array(labels)
    
    def run(self):
        iters = 0
        while True:
            labels = self.get_labels(self.X, self.centroids)
            new_centroids = np.zeros_like(self.centroids)
            
            for i in range(self.k):
                points = self.X[labels == i]
                if len(points) == 0:
                    new_centroids[i] = self.centroids[i]
                else:
                    new_centroids[i] = np.mean(points, axis=0)
            
            if np.allclose(self.centroids, new_centroids) or iters == self.max_iters:
                break
                
            self.centroids = new_centroids
            iters += 1
            
        final_labels = self.get_labels(self.X, self.centroids)
        return final_labels