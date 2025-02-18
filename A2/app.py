from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import multivariate_normal
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class EMModel:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit_predict(self, X):
        """Fit the EM model and return cluster assignments"""
        self.X = X
        self.N, self.dim = X.shape
        
        
        indices = np.random.choice(self.N, self.k, replace=False)
        self.mu = self.X[indices]
        self.pi = np.ones(self.k) / self.k
        self.sigma = np.array([np.eye(self.dim) * 5.0 for _ in range(self.k)])
        
        for _ in range(self.max_iters):
            
            r = self._expectation()
            
            
            self._maximization(r)
        
        r = self._expectation()
        labels = r.argmax(axis=1)
        
        return labels, self.mu, self.sigma, self.pi
    
    def _expectation(self):
        """E-step: compute responsibilities"""
        r = np.zeros((self.N, self.k))
        
        for c in range(self.k):
            try:
                dist = multivariate_normal(
                    mean=self.mu[c], 
                    cov=self.sigma[c], 
                    allow_singular=True
                )
                r[:, c] = self.pi[c] * dist.pdf(self.X)
            except Exception as e:
                logger.error(f"Error in _expectation for cluster {c}: {str(e)}")
                raise
            
       
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum[r_sum == 0] = 1e-10  # avoid division by zero
        r = r / r_sum
        return r
    
    def _maximization(self, r):
        """M-step: update parameters"""
        
        r_sum = r.sum(axis=0)
        r_sum[r_sum == 0] = 1e-10  
        self.mu = r.T @ self.X / r_sum[:, np.newaxis]
        
       
        self.pi = r_sum / self.N
       
        for c in range(self.k):
            diff = self.X - self.mu[c]
            self.sigma[c] = (r[:, c:c+1] * diff).T @ diff / r_sum[c]
           
            self.sigma[c] += np.eye(self.dim) * 1e-6

def load_and_preprocess_data():
    """Load and preprocess the student data"""
    try:
        data = pd.read_csv('PA2_data.csv')
        X = data.iloc[:, 1:].to_numpy()  
        X = X / X.max(axis=0)  
        return X
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get PCA transformed data"""
    try:
        X = load_and_preprocess_data()
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X)
        return jsonify({
            'data': pca_data.tolist()
        })
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    """Perform clustering using specified algorithm"""
    try:
        data = request.json
        X = np.array(data['data'])
        method = data.get('method', 'kmeans')
        n_clusters = data.get('n_clusters', 3)
        
        logger.info(f"Clustering method: {method}, n_clusters: {n_clusters}")
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            return jsonify({
                'labels': labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist()
            })
            
        elif method == 'em':
            em = EMModel(k=n_clusters)
            labels, centroids, covariances, weights = em.fit_predict(X)
            return jsonify({
                'labels': labels.tolist(),
                'centroids': centroids.tolist(),
                'covariances': covariances.tolist(),
                'weights': weights.tolist()
            })
            
        elif method == 'hierarchical':
            Z = linkage(X, method='ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
            return jsonify({
                'labels': labels.tolist(),
                'linkage_matrix': Z.tolist()
            })
            
        else:
            return jsonify({'error': 'Invalid clustering method'}), 400
            
    except Exception as e:
        logger.error(f"Error in cluster_data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)