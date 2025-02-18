import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

class DataProcessor:
    @staticmethod
    def apply_pca(X, n_components=2):
        """Apply PCA transformation to the data"""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

class KMeansModel:
    def __init__(self, n_clusters=3, max_iters=100):
        self.k = n_clusters
        self.max_iters = max_iters
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iters)
    
    def fit_predict(self, X):
        """Fit model and return cluster assignments and centroids"""
        labels = self.model.fit_predict(X)
        return {
            'labels': labels,
            'centroids': self.model.cluster_centers_
        }

class EMModel:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit_predict(self, X):
        """Fit the EM model and return cluster assignments"""
        self.X = X
        self.N, self.dim = X.shape
        self.mu = self.X[np.random.choice(self.N, self.k, replace=False)]
        self.pi = np.ones(self.k) / self.k
        self.sigma = np.array([np.eye(self.dim) * 5.0 for _ in range(self.k)])
        
        for _ in range(self.max_iters):
            r = self._expectation()
 
            self._maximization(r)
        
        r = self._expectation()
        labels = r.argmax(axis=1)
        
        return {
            'labels': labels,
            'centroids': self.mu,
            'covariances': self.sigma,
            'weights': self.pi
        }
    
    def _expectation(self):
        """E-step: compute responsibilities"""
        r = np.zeros((self.N, self.k))
        
        for c in range(self.k):
            prob = multivariate_normal(
                mean=self.mu[c], 
                cov=self.sigma[c], 
                allow_singular=True
            ).pdf(self.X)
            r[:, c] = self.pi[c] * prob
        r = r / np.sum(r, axis=1, keepdims=True)
        return r
    
    def _maximization(self, r):
        """M-step: update parameters"""
        self.mu = r.T @ self.X / r.sum(axis=0)[:, np.newaxis]
        
        self.pi = r.sum(axis=0) / self.N
        for c in range(self.k):
            diff = self.X - self.mu[c]
            weight = r[:, c, np.newaxis] * diff
            self.sigma[c] = weight.T @ diff / r[:, c].sum()

class HierarchicalModel:
    def __init__(self, method='ward'):
        self.method = method
    
    def fit_predict(self, X, n_clusters=3):
        """Perform hierarchical clustering and return linkage matrix"""
        self.linkage_matrix = linkage(X, method=self.method)
        
       
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        return {
            'labels': labels,
            'linkage_matrix': self.linkage_matrix
        }