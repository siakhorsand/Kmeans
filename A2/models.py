import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

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
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iters, random_state=42)
    
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
        
        # Initialize parameters
        indices = np.random.choice(self.N, self.k, replace=False)
        self.mu = self.X[indices]
        self.pi = np.ones(self.k) / self.k
        self.sigma = np.array([np.eye(self.dim) * 0.5 for _ in range(self.k)])
        
        for _ in range(self.max_iters):
            # E-step
            r = self._expectation()
            
            # M-step
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
            try:
                prob = multivariate_normal(
                    mean=self.mu[c], 
                    cov=self.sigma[c], 
                    allow_singular=True
                ).pdf(self.X)
                r[:, c] = self.pi[c] * prob
            except Exception as e:
                logger.error(f"Error in EM expectation step for cluster {c}: {str(e)}")
                r[:, c] = self.pi[c] * 1e-10
                
        # Normalize responsibilities
        r_sum = np.sum(r, axis=1, keepdims=True)
        r_sum[r_sum == 0] = 1e-10  # Avoid division by zero
        r = r / r_sum
        
        return r
    
    def _maximization(self, r):
        """M-step: update parameters"""
        # Update weights
        r_sum = r.sum(axis=0)
        r_sum[r_sum == 0] = 1e-10  # Avoid division by zero
        self.pi = r_sum / self.N
        
        # Update means
        self.mu = r.T @ self.X / r_sum[:, np.newaxis]
        
        # Update covariances
        for c in range(self.k):
            diff = self.X - self.mu[c]
            weighted_diff = r[:, c, np.newaxis] * diff
            self.sigma[c] = weighted_diff.T @ diff / r_sum[c]
            
            # Add regularization to avoid singular matrices
            self.sigma[c] += np.eye(self.dim) * 1e-6

class HierarchicalModel:
    def __init__(self, method='ward'):
        self.method = method
    
    def fit_predict(self, X, n_clusters=3):
        """Perform hierarchical clustering and return linkage matrix"""
        self.linkage_matrix = linkage(X, method=self.method)
        
        # Get cluster labels
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        return {
            'labels': labels,
            'linkage_matrix': self.linkage_matrix
        }