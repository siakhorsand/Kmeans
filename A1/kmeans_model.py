import numpy as np

class KmeansModel:
    def __init__(self, X, k, max_iters=100):
        self.X = X
        self.k = k
        self.max_iters = max_iters
        self.dim = X.shape[1]
        self.N = X.shape[0]

        indices = np.random.choice(self.N, self.k, replace=False)
        self.centroids = self.X[indices]
    
    def get_labels(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

        return np.argmin(distances, axis=1)
    
    def run(self):
        prev_centroids = None
        iters = 0
        
        while iters < self.max_iters:

            labels = self.get_labels(self.X, self.centroids)
            

            new_centroids = np.array([
                self.X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 
                else self.centroids[k] 
                for k in range(self.k)
            ])

            if prev_centroids is not None and np.allclose(prev_centroids, new_centroids):
                break
                
            prev_centroids = new_centroids.copy()
            self.centroids = new_centroids
            iters += 1
            
        return self.get_labels(self.X, self.centroids)